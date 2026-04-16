package com.example.dps_zn

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.graphics.YuvImage
import android.os.SystemClock
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import java.io.BufferedReader
import java.io.ByteArrayOutputStream
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Locale
import java.util.concurrent.ConcurrentHashMap
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/**
 * Анализ кадра: TFLite Support, YOLOv8, телеметрия.
 * Модель грузится лениво при первом кадре (не блокирует UI).
 */
class ImageAnalyzer(
    context: Context,
    private val onResults: (imageWidth: Int, imageHeight: Int, detections: List<GraphicOverlayView.OverlayDetection>) -> Unit
) : ImageAnalysis.Analyzer {

    private val appContext = context.applicationContext

    private val initLock = Any()
    private var interpreter: Interpreter? = null
    private var labelsAll: List<String> = emptyList()
    private var labelsForModel: List<String> = emptyList()
    private var imageProcessor: ImageProcessor? = null
    private var inputIsUint8: Boolean = false
    private var outputTotal: Int = 0
    private var outputShape: IntArray = intArrayOf()
    private var outputDataType: DataType = DataType.FLOAT32
    private var outputQuantScale: Float = 1f
    private var outputQuantZeroPoint: Int = 0
    private lateinit var outputFloatBuffer: FloatArray
    private var outputRawBuffer: ByteBuffer? = null
    private var outputQuantBytes: ByteArray? = null
    private var inputBufferDirect: ByteBuffer? = null
    private var outputClassCount: Int = 0

    private val ignoreLabels = setOf("null")
    private val lastTelemetryAtMs = ConcurrentHashMap<String, Long>()
    private val telemetryCooldownMs = 10_000L
    private val letterboxPaint = Paint(Paint.FILTER_BITMAP_FLAG)

    @Volatile
    private var loggedTensorMeta = false
    @Volatile
    private var loggedFrameMeta = false
    private var analyzedFrameIndex = 0L
    private var detectionTracks: List<TrackedBox> = emptyList()

    private fun ensureEngine() {
        if (interpreter != null) return
        synchronized(initLock) {
            if (interpreter != null) return
            val model = FileUtil.loadMappedFile(appContext, "best_int8.tflite")
            val interp = Interpreter(
                model,
                Interpreter.Options().apply { setNumThreads(4) }
            )
            val lbls = appContext.assets.open("labels.txt").use { stream ->
                BufferedReader(InputStreamReader(stream)).lineSequence()
                    .map { it.trim() }
                    .filter { it.isNotEmpty() }
                    .map { normalizeLabel(it) }
                    .toList()
            }
            val inTensor = interp.getInputTensor(0)
            val uint8 = inTensor.dataType() == DataType.UINT8
            val proc = ImageProcessor.Builder()
                .build()
            val outTensor = interp.getOutputTensor(0)
            val shape = outTensor.shape()
            val classCount = inferClassCount(shape)
            val modelLbls = buildModelLabels(lbls, classCount)
            val inQuant = inTensor.quantizationParams()
            val outQuant = outTensor.quantizationParams()
            Log.d(TAG_DEBUG, "Labels size: ${lbls.size}, first label: ${lbls.getOrNull(0)}")
            Log.d(TAG_DEBUG, "Model classes: $classCount, labels used: $modelLbls")
            if (ENABLE_DEBUG_LOGS) {
                Log.i(
                    TAG_DEBUG,
                    "input shape=${inTensor.shape().contentToString()} dtype=${inTensor.dataType()} " +
                        "qScale=${inQuant.scale} qZero=${inQuant.zeroPoint}"
                )
                Log.i(
                    TAG_DEBUG,
                    "output shape=${shape.contentToString()} dtype=${outTensor.dataType()} " +
                        "qScale=${outQuant.scale} qZero=${outQuant.zeroPoint} " +
                        "outputs=${interp.outputTensorCount} labelsRaw=${lbls.size} labelsActive=${modelLbls.size}"
                )
            }
            val total = shape.fold(1, Int::times)
            val inBytes = inTensor.numBytes()
            val inBuf = ByteBuffer.allocateDirect(inBytes).order(ByteOrder.nativeOrder())
            val outType = outTensor.dataType()
            val outBuf = ByteBuffer.allocateDirect(outTensor.numBytes()).order(ByteOrder.nativeOrder())

            interpreter = interp
            labelsAll = lbls
            labelsForModel = modelLbls
            imageProcessor = proc
            inputIsUint8 = uint8
            outputShape = shape
            outputTotal = total
            outputDataType = outType
            outputQuantScale = outQuant.scale
            outputQuantZeroPoint = outQuant.zeroPoint
            outputClassCount = classCount
            outputFloatBuffer = FloatArray(total)
            outputRawBuffer = outBuf
            outputQuantBytes =
                if (outType == DataType.UINT8 || outType == DataType.INT8) ByteArray(outTensor.numBytes()) else null
            inputBufferDirect = inBuf
        }
    }

    override fun analyze(imageProxy: ImageProxy) {
        try {
            ensureEngine()
            analyzedFrameIndex += 1L
            if (analyzedFrameIndex % INFERENCE_EVERY_N_FRAMES != 0L) return
            val interp = interpreter ?: return
            val inBuf = inputBufferDirect ?: return
            val proc = imageProcessor ?: return

            val rotation = imageProxy.imageInfo.rotationDegrees
            val rawBitmap = imageProxyToBitmap(imageProxy) ?: return
            if (rawBitmap.isRecycled) {
                Log.w(TAG_DEBUG, "skip frame: raw bitmap is recycled")
                return
            }
            val oriented = rotateIfNeeded(rawBitmap, rotation)
            if (oriented.isRecycled) {
                Log.w(TAG_DEBUG, "skip frame: oriented bitmap is recycled")
                return
            }

            val rw = oriented.width
            val rh = oriented.height
            if (ENABLE_DEBUG_LOGS && !loggedFrameMeta) {
                loggedFrameMeta = true
                Log.i(
                    TAG_DEBUG,
                    "frame image=${imageProxy.width}x${imageProxy.height} rotation=$rotation oriented=${rw}x${rh}"
                )
            }

            val letterbox = letterboxBitmap(oriented)
            val tensorImage = if (inputIsUint8) TensorImage(DataType.UINT8) else TensorImage(DataType.FLOAT32)
            tensorImage.load(letterbox.bitmap)
            if (letterbox.bitmap.isRecycled) {
                Log.w(TAG_DEBUG, "skip frame: bitmap became recycled before process")
                return
            }

            val processed = proc.process(tensorImage)
            if (!inputIsUint8) {
                normalizeFloatBufferToUnitRange(processed.buffer)
            }
            feedInput(processed, inBuf)

            val rawOutput = runInference(interp, inBuf)
            val parsed = when {
                (outputDataType == DataType.UINT8 || outputDataType == DataType.INT8) && rawOutput.quant != null ->
                    parseYoloV8Quant(rawOutput.quant, outputShape)
                rawOutput.float != null ->
                    parseYoloV8(rawOutput.float, outputShape)
                else -> ParseResult(emptyList(), 0f)
            }
            val stableBoxes = stabilizeDetections(parsed.boxes)
            if (ENABLE_DEBUG_LOGS) {
                Log.d(
                    TAG_DEBUG,
                    "frameMaxScore=${String.format(Locale.US, "%.4f", parsed.maxScore)} " +
                        "boxesAfterNms=${parsed.boxes.size} stable=${stableBoxes.size}"
                )
            }
            val overlayItems = ArrayList<GraphicOverlayView.OverlayDetection>()
            for (b in stableBoxes) {
                val nameRaw = labelsForModel.getOrNull(b.classIndex)
                    ?: labelsAll.getOrNull(b.classIndex)
                    ?: "class_${b.classIndex}"
                val name = normalizeLabel(nameRaw)
                if (name.lowercase(Locale.ROOT) in ignoreLabels || name.isBlank()) continue
                val rect = modelRectToImageRect(b.rect, letterbox)
                if (rect.width() <= 1f || rect.height() <= 1f) continue
                overlayItems.add(GraphicOverlayView.OverlayDetection(name, b.score, rect))
                maybeSendTelemetry(name, b.score)
            }
            onResults(rw, rh, overlayItems)
        } catch (t: Throwable) {
            Log.e(TAG, "analyze failed", t)
        } finally {
            imageProxy.close()
        }
    }

    private data class InferenceOutput(
        val float: FloatArray? = null,
        val quant: ByteArray? = null
    )

    private data class LetterboxFrame(
        val bitmap: Bitmap,
        val scale: Float,
        val dx: Float,
        val dy: Float,
        val sourceWidth: Int,
        val sourceHeight: Int
    )

    private fun runInference(interp: Interpreter, input: ByteBuffer): InferenceOutput {
        input.rewind()
        val rawOut = outputRawBuffer ?: return InferenceOutput(float = outputFloatBuffer)
        rawOut.rewind()
        interp.run(input, rawOut)
        rawOut.rewind()

        return when (outputDataType) {
            DataType.FLOAT32 -> {
                outputFloatBuffer.fill(0f)
                rawOut.asFloatBuffer().get(outputFloatBuffer, 0, outputTotal)
                InferenceOutput(float = outputFloatBuffer)
            }
            DataType.UINT8, DataType.INT8 -> {
                val quant = outputQuantBytes ?: ByteArray(rawOut.remaining()).also { outputQuantBytes = it }
                rawOut.get(quant, 0, min(quant.size, rawOut.remaining()))
                rawOut.rewind()
                InferenceOutput(quant = quant)
            }
            else -> {
                outputFloatBuffer.fill(0f)
                rawOut.asFloatBuffer().get(outputFloatBuffer, 0, outputTotal)
                InferenceOutput(float = outputFloatBuffer)
            }
        }
    }

    private fun normalizeFloatBufferToUnitRange(buffer: ByteBuffer) {
        buffer.rewind()
        val fb = buffer.asFloatBuffer()
        val n = fb.remaining()
        if (n == 0) return
        val arr = FloatArray(n)
        fb.get(arr)
        for (i in arr.indices) arr[i] /= 255f
        buffer.rewind()
        buffer.asFloatBuffer().put(arr)
        buffer.rewind()
    }

    private fun feedInput(processed: TensorImage, inputBufferDirect: ByteBuffer) {
        val buffer = processed.buffer
        buffer.rewind()
        inputBufferDirect.rewind()
        val lim = min(buffer.remaining(), inputBufferDirect.remaining())
        val tmp = ByteArray(lim)
        buffer.get(tmp)
        inputBufferDirect.put(tmp)
        inputBufferDirect.rewind()
    }

    private fun maybeSendTelemetry(label: String, conf: Float) {
        if (!ENABLE_TELEMETRY) return
        try {
            val now = SystemClock.elapsedRealtime()
            val prev = lastTelemetryAtMs[label] ?: 0L
            if (now - prev < telemetryCooldownMs) return
            lastTelemetryAtMs[label] = now
            FirebaseTelemetryManager.sendSignData(label, conf.toDouble())
        } catch (e: Exception) {
            Log.w(TAG, "telemetry: $e")
        }
    }

    private fun inferClassCount(shape: IntArray): Int {
        if (shape.size != 3 || shape[0] != 1) return 0
        val dim1 = shape[1]
        val dim2 = shape[2]
        return when {
            dim1 >= YOLO_CLASS_START_ROW && dim2 > dim1 -> dim1 - YOLO_CLASS_START_ROW
            dim2 >= YOLO_CLASS_START_ROW && dim1 > dim2 -> dim2 - YOLO_CLASS_START_ROW
            else -> 0
        }.coerceAtLeast(0)
    }

    private fun buildModelLabels(labels: List<String>, classCount: Int): List<String> {
        if (classCount <= 0) return labels
        val active = labels
            .filter { it.lowercase(Locale.ROOT) !in ignoreLabels }
            .distinct()
        return active.take(classCount)
    }

    private fun modelClassCount(): Int {
        return if (outputClassCount > 0) outputClassCount else YOLO_NUM_CLASSES
    }

    private fun isIgnoredClass(classIndex: Int): Boolean {
        val label = labelsForModel.getOrNull(classIndex)?.lowercase(Locale.ROOT) ?: return false
        return label in ignoreLabels
    }

    private data class Box640(val rect: RectF, val classIndex: Int, val score: Float)
    private data class TrackedBox(
        val box: Box640,
        val hits: Int,
        val classVotes: Map<Int, Int>,
        val classScores: Map<Int, Float>
    )
    private data class ParseResult(val boxes: List<Box640>, val maxScore: Float)

    /**
     * Поддержка типичных экспортов YOLOv8 TFLite:
     * - [1, 4+nc, A] или [1, A, 4+nc] (nc из модели, не из длины labels.txt)
     * - [1, N, 6] / [1, 6, N] — xyxy + score + class (после встроенного NMS)
     */
    private fun stabilizeDetections(current: List<Box640>): List<Box640> {
        if (current.isEmpty()) {
            detectionTracks = emptyList()
            return emptyList()
        }
        val previous = detectionTracks
        val next = ArrayList<TrackedBox>()
        val stable = ArrayList<Box640>()
        for (box in current) {
            val match = previous
                .maxByOrNull { iou(it.box.rect, box.rect) }
            val hits = if (match != null && iou(match.box.rect, box.rect) >= STABILITY_IOU_THRESHOLD) {
                match.hits + 1
            } else {
                1
            }
            val votes = match?.classVotes?.toMutableMap() ?: mutableMapOf()
            val scores = match?.classScores?.toMutableMap() ?: mutableMapOf()
            votes[box.classIndex] = (votes[box.classIndex] ?: 0) + 1
            scores[box.classIndex] = max(scores[box.classIndex] ?: 0f, box.score)
            val stableClass = votes.maxWithOrNull(
                compareBy<Map.Entry<Int, Int>> { it.value }
                    .thenBy { scores[it.key] ?: 0f }
            )?.key ?: box.classIndex
            val stableScore = scores[stableClass] ?: box.score
            val trackedBox = Box640(RectF(box.rect), stableClass, stableScore)
            next.add(TrackedBox(trackedBox, hits, votes, scores))
            if (hits >= STABLE_FRAMES_REQUIRED) stable.add(trackedBox)
        }
        detectionTracks = next
        return stable
    }

    private fun parseYoloV8(flat: FloatArray, shape: IntArray): ParseResult {
        if (ENABLE_DEBUG_LOGS && !loggedTensorMeta) {
            loggedTensorMeta = true
            Log.i(TAG_DEBUG, "parseYoloV8 shape=${shape.contentToString()} flat=${flat.size}")
        }
        if (shape.isEmpty() || shape[0] != 1) return ParseResult(emptyList(), 0f)

        if (shape.size == 3) {
            val a = shape[1]
            val b = shape[2]
            when {
                b == 6 && a > 6 -> return parseRowsNx6(flat, a)
                a == 6 && b > 6 -> return parseRows6xN(flat, b)
            }
        }
        if (shape.size != 3) return ParseResult(emptyList(), 0f)

        val dim1 = shape[1]
        val dim2 = shape[2]
        return when {
            dim1 == YOLO_FEATURE_ROWS && dim2 == YOLO_ANCHORS ->
                parseYoloNineRowMajor(flat, YOLO_ANCHORS)
            dim1 == YOLO_ANCHORS && dim2 == YOLO_FEATURE_ROWS ->
                parseYoloNineAnchorMajor(flat, YOLO_ANCHORS)
            else -> {
                Log.w(TAG, "Неизвестная форма выхода: [1,$dim1,$dim2], labels=${labelsForModel.size}")
                ParseResult(emptyList(), 0f)
            }
        }
    }

    private fun parseYoloV8Quant(flat: ByteArray, shape: IntArray): ParseResult {
        if (ENABLE_DEBUG_LOGS && !loggedTensorMeta) {
            loggedTensorMeta = true
            Log.i(TAG_DEBUG, "parseYoloV8 shape=${shape.contentToString()} quant=${flat.size}")
        }
        if (shape.size != 3 || shape[0] != 1) return ParseResult(emptyList(), 0f)
        val dim1 = shape[1]
        val dim2 = shape[2]
        return when {
            dim1 == YOLO_FEATURE_ROWS && dim2 == YOLO_ANCHORS ->
                parseYoloNineQuantByAnchors(YOLO_ANCHORS) { row, anchor ->
                    val index = row * YOLO_ANCHORS + anchor
                    if (index in flat.indices) quantToRawInt(flat[index]) else 0
                }
            dim1 == YOLO_ANCHORS && dim2 == YOLO_FEATURE_ROWS ->
                parseYoloNineQuantByAnchors(YOLO_ANCHORS) { row, anchor ->
                    val index = anchor * YOLO_FEATURE_ROWS + row
                    if (index in flat.indices) quantToRawInt(flat[index]) else 0
                }
            else -> {
                Log.w(TAG, "РќРµРёР·РІРµСЃС‚РЅР°СЏ С„РѕСЂРјР° РІС‹С…РѕРґР° quant: [1,$dim1,$dim2]")
                ParseResult(emptyList(), 0f)
            }
        }
    }

    private fun parseYoloNineRowMajor(
        flat: FloatArray,
        numAnchors: Int
    ): ParseResult {
        return parseYoloNineByAnchors(numAnchors) { row, anchor ->
            val index = row * numAnchors + anchor
            if (index in flat.indices) flat[index] else 0f
        }
    }

    private fun parseYoloNineAnchorMajor(
        flat: FloatArray,
        numAnchors: Int
    ): ParseResult {
        return parseYoloNineByAnchors(numAnchors) { row, anchor ->
            val index = anchor * YOLO_FEATURE_ROWS + row
            if (index in flat.indices) flat[index] else 0f
        }
    }

    private fun parseYoloNineByAnchors(
        numAnchors: Int,
        valueAt: (row: Int, anchor: Int) -> Float
    ): ParseResult {
        val raw = ArrayList<Triple<RectF, Int, Float>>()
        var maxScoreSeen = 0f
        val classCount = modelClassCount()
        for (anchor in 0 until numAnchors) {
            var bestCls = 0
            var bestScore = 0f
            var secondBestScore = 0f
            for (c in 0 until classCount) {
                val row = YOLO_CLASS_START_ROW + c
                val score = toClassProb(valueAt(row, anchor))
                if (score > bestScore) {
                    secondBestScore = bestScore
                    bestScore = score
                    bestCls = c
                } else if (score > secondBestScore) {
                    secondBestScore = score
                }
            }
            if (bestScore > maxScoreSeen) maxScoreSeen = bestScore
            if (bestScore <= CONF_THRESHOLD) continue
            if (bestScore - secondBestScore < CLASS_MARGIN_THRESHOLD) continue
            if (isIgnoredClass(bestCls)) continue

            val xCenter = valueAt(0, anchor)
            val yCenter = valueAt(1, anchor)
            val w = valueAt(2, anchor)
            val h = valueAt(3, anchor)
            val rect = decodeBox640Strict(xCenter, yCenter, w, h)
            if (rect.width() <= 1f || rect.height() <= 1f) continue
            raw.add(Triple(rect, bestCls, bestScore))
        }
        return ParseResult(nms(raw), maxScoreSeen)
    }

    private fun parseYoloNineQuantByAnchors(
        numAnchors: Int,
        rawAt: (row: Int, anchor: Int) -> Int
    ): ParseResult {
        val raw = ArrayList<Triple<RectF, Int, Float>>()
        var maxScoreSeen = 0f
        val classCount = modelClassCount()
        for (anchor in 0 until numAnchors) {
            var bestCls = 0
            var bestScore = 0f
            var secondBestScore = 0f
            for (c in 0 until classCount) {
                val row = YOLO_CLASS_START_ROW + c
                val score = toClassProb(dequantizeQuantValue(rawAt(row, anchor)))
                if (score > bestScore) {
                    secondBestScore = bestScore
                    bestScore = score
                    bestCls = c
                } else if (score > secondBestScore) {
                    secondBestScore = score
                }
            }
            if (bestScore > maxScoreSeen) maxScoreSeen = bestScore
            if (bestScore <= CONF_THRESHOLD) continue
            if (bestScore - secondBestScore < CLASS_MARGIN_THRESHOLD) continue
            if (isIgnoredClass(bestCls)) continue

            val xCenter = dequantizeQuantValue(rawAt(0, anchor))
            val yCenter = dequantizeQuantValue(rawAt(1, anchor))
            val w = dequantizeQuantValue(rawAt(2, anchor))
            val h = dequantizeQuantValue(rawAt(3, anchor))
            val rect = decodeBox640Strict(xCenter, yCenter, w, h)
            if (rect.width() <= 1f || rect.height() <= 1f) continue
            raw.add(Triple(rect, bestCls, bestScore))
        }
        return ParseResult(nms(raw), maxScoreSeen)
    }

    /** [1, N, 6]: x1,y1,x2,y2,score,class */
    private fun parseRowsNx6(flat: FloatArray, n: Int): ParseResult {
        val raw = ArrayList<Triple<RectF, Int, Float>>()
        var maxScoreSeen = 0f
        for (i in 0 until n) {
            val o = i * 6
            if (o + 5 >= flat.size) break
            val x1 = flat[o]
            val y1 = flat[o + 1]
            val x2 = flat[o + 2]
            val y2 = flat[o + 3]
            val score = toClassProb(flat[o + 4])
            if (score > maxScoreSeen) maxScoreSeen = score
            val cls = flat[o + 5].toInt().coerceAtLeast(0)
            if (score < CONF_THRESHOLD) continue
            if (x2 <= x1 || y2 <= y1) continue
            raw.add(Triple(RectF(x1, y1, x2, y2), cls, score))
        }
        return ParseResult(nms(raw), maxScoreSeen)
    }

    /** [1, 6, N]: по строкам признаков */
    private fun parseRows6xN(flat: FloatArray, n: Int): ParseResult {
        val raw = ArrayList<Triple<RectF, Int, Float>>()
        var maxScoreSeen = 0f
        for (i in 0 until n) {
            val x1 = flat[0 * n + i]
            val y1 = flat[1 * n + i]
            val x2 = flat[2 * n + i]
            val y2 = flat[3 * n + i]
            val score = toClassProb(flat[4 * n + i])
            if (score > maxScoreSeen) maxScoreSeen = score
            val cls = flat[5 * n + i].toInt().coerceAtLeast(0)
            if (score < CONF_THRESHOLD) continue
            if (x2 <= x1 || y2 <= y1) continue
            raw.add(Triple(RectF(x1, y1, x2, y2), cls, score))
        }
        return ParseResult(nms(raw), maxScoreSeen)
    }

    private fun quantToRawInt(v: Byte): Int {
        return if (outputDataType == DataType.UINT8) v.toInt() and 0xFF else v.toInt()
    }

    private fun dequantizeQuantValue(raw: Int): Float {
        return (raw - outputQuantZeroPoint) * outputQuantScale
    }

    private fun toClassProb(raw: Float): Float {
        if (!raw.isFinite()) return 0f
        if (raw in 0f..1f) return raw
        return sigmoid(raw)
    }

    private fun normalizeLabel(raw: String): String = raw.trim().trimEnd('\\')

    private fun letterboxBitmap(src: Bitmap): LetterboxFrame {
        val scale = min(MODEL_SIZE / src.width.toFloat(), MODEL_SIZE / src.height.toFloat())
        val scaledWidth = src.width * scale
        val scaledHeight = src.height * scale
        val dx = (MODEL_SIZE - scaledWidth) / 2f
        val dy = (MODEL_SIZE - scaledHeight) / 2f
        val out = Bitmap.createBitmap(MODEL_SIZE, MODEL_SIZE, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(out)
        canvas.drawColor(Color.BLACK)
        canvas.drawBitmap(src, null, RectF(dx, dy, dx + scaledWidth, dy + scaledHeight), letterboxPaint)
        return LetterboxFrame(out, scale, dx, dy, src.width, src.height)
    }

    private fun modelRectToImageRect(rect: RectF, frame: LetterboxFrame): RectF {
        val left = ((rect.left - frame.dx) / frame.scale).coerceIn(0f, frame.sourceWidth.toFloat())
        val top = ((rect.top - frame.dy) / frame.scale).coerceIn(0f, frame.sourceHeight.toFloat())
        val right = ((rect.right - frame.dx) / frame.scale).coerceIn(0f, frame.sourceWidth.toFloat())
        val bottom = ((rect.bottom - frame.dy) / frame.scale).coerceIn(0f, frame.sourceHeight.toFloat())
        return RectF(left, top, right, bottom)
    }

    private fun decodeBox640(p0: Float, p1: Float, p2: Float, p3: Float): RectF {
        // Для head YOLOv8 [1, 4+nc, A] ожидаем cx,cy,w,h.
        var cx = p0
        var cy = p1
        var w = p2
        var h = p3
        if (maxOf(cx, cy, w, h) <= 2f) {
            cx *= MODEL_SIZE
            cy *= MODEL_SIZE
            w *= MODEL_SIZE
            h *= MODEL_SIZE
        }
        val halfW = w / 2f
        val halfH = h / 2f
        return RectF(
            (cx - halfW).coerceIn(0f, MODEL_SIZE.toFloat()),
            (cy - halfH).coerceIn(0f, MODEL_SIZE.toFloat()),
            (cx + halfW).coerceIn(0f, MODEL_SIZE.toFloat()),
            (cy + halfH).coerceIn(0f, MODEL_SIZE.toFloat())
        )
    }

    private fun decodeBox640Strict(p0: Float, p1: Float, p2: Float, p3: Float): RectF {
        var cx = p0
        var cy = p1
        var w = p2
        var h = p3
        if (!cx.isFinite() || !cy.isFinite() || !w.isFinite() || !h.isFinite()) return RectF()
        if (maxOf(cx, cy, w, h) <= 2f) {
            cx *= MODEL_SIZE
            cy *= MODEL_SIZE
            w *= MODEL_SIZE
            h *= MODEL_SIZE
        }
        if (w <= 0f || h <= 0f) return RectF()
        val halfW = w / 2f
        val halfH = h / 2f
        val left = cx - halfW
        val top = cy - halfH
        val right = cx + halfW
        val bottom = cy + halfH
        if (right <= 0f || bottom <= 0f || left >= MODEL_SIZE || top >= MODEL_SIZE) return RectF()
        return RectF(
            left.coerceIn(0f, MODEL_SIZE.toFloat()),
            top.coerceIn(0f, MODEL_SIZE.toFloat()),
            right.coerceIn(0f, MODEL_SIZE.toFloat()),
            bottom.coerceIn(0f, MODEL_SIZE.toFloat())
        )
    }

    private fun nms(
        boxes: List<Triple<RectF, Int, Float>>,
        iouTh: Float = NMS_IOU_THRESHOLD,
        limit: Int = MAX_DETECTIONS
    ): List<Box640> {
        val sorted = boxes.sortedByDescending { it.third }
        val kept = ArrayList<Box640>()
        val perClassCount = HashMap<Int, Int>()
        for (t in sorted) {
            if (kept.size >= limit) break
            val r = t.first
            val area = r.width() * r.height()
            if (r.width() < MIN_BOX_SIZE_PX || r.height() < MIN_BOX_SIZE_PX) continue
            if (area > MODEL_SIZE * MODEL_SIZE * MAX_BOX_AREA_RATIO) continue
            val aspect = r.width() / r.height()
            if (aspect < MIN_BOX_ASPECT_RATIO || aspect > MAX_BOX_ASPECT_RATIO) continue
            val cls = t.second
            val clsCount = perClassCount[cls] ?: 0
            if (clsCount >= MAX_BOXES_PER_CLASS) continue
            var ok = true
            for (k in kept) {
                if (iou(r, k.rect) > iouTh) {
                    ok = false
                    break
                }
            }
            if (ok) {
                kept.add(Box640(RectF(r), cls, t.third))
                perClassCount[cls] = clsCount + 1
            }
        }
        return kept
    }

    private fun iou(a: RectF, b: RectF): Float {
        val interLeft = max(a.left, b.left)
        val interTop = max(a.top, b.top)
        val interRight = min(a.right, b.right)
        val interBottom = min(a.bottom, b.bottom)
        val iw = (interRight - interLeft).coerceAtLeast(0f)
        val ih = (interBottom - interTop).coerceAtLeast(0f)
        val inter = iw * ih
        val u = a.width() * a.height() + b.width() * b.height() - inter
        return if (u <= 0f) 0f else inter / u
    }

    private fun sigmoid(x: Float): Float =
        (1.0 / (1.0 + exp(-x.toDouble()))).toFloat()

    private fun rotateIfNeeded(src: Bitmap, rotationDegrees: Int): Bitmap {
        if (rotationDegrees == 0) return src
        val m = Matrix()
        m.postRotate(rotationDegrees.toFloat())
        return Bitmap.createBitmap(src, 0, 0, src.width, src.height, m, true)
    }

    private fun imageProxyToBitmap(image: ImageProxy): Bitmap? {
        return if (image.planes.size == 1) {
            rgbaImageProxyToBitmap(image)
        } else {
            yuv420888ToBitmap(image)
        }
    }

    private fun rgbaImageProxyToBitmap(image: ImageProxy): Bitmap? {
        val plane = image.planes[0]
        val buffer = plane.buffer.duplicate()
        buffer.rewind()
        val pixelStride = plane.pixelStride
        val rowStride = plane.rowStride
        val width = image.width
        val height = image.height
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        if (pixelStride == 4 && rowStride == width * 4) {
            bitmap.copyPixelsFromBuffer(buffer)
            return bitmap
        }
        val rowPixels = IntArray(width)
        for (y in 0 until height) {
            for (x in 0 until width) {
                val r = buffer.get().toInt() and 0xFF
                val g = buffer.get().toInt() and 0xFF
                val b = buffer.get().toInt() and 0xFF
                val a = buffer.get().toInt() and 0xFF
                rowPixels[x] = (a shl 24) or (r shl 16) or (g shl 8) or b
            }
            val rowPadding = rowStride - width * pixelStride
            if (rowPadding > 0) buffer.position(buffer.position() + rowPadding)
            bitmap.setPixels(rowPixels, 0, width, 0, y, width, 1)
        }
        return bitmap
    }

    private fun yuv420888ToBitmap(image: ImageProxy): Bitmap? {
        val nv21 = yuv420888ToNv21(image) ?: return null
        val yuv = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
        val out = ByteArrayOutputStream()
        yuv.compressToJpeg(Rect(0, 0, image.width, image.height), 85, out)
        val bytes = out.toByteArray()
        return BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
    }

    private fun yuv420888ToNv21(image: ImageProxy): ByteArray? {
        val width = image.width
        val height = image.height
        val ySize = width * height
        val uvSize = width * height / 4
        val out = ByteArray(ySize + uvSize * 2)
        val yPlane = image.planes[0]
        copyPlaneRowWise(yPlane, width, height, out, 0, 1)
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]
        val chromaHeight = height / 2
        val chromaWidth = width / 2
        val uBuffer = uPlane.buffer.duplicate()
        val vBuffer = vPlane.buffer.duplicate()
        uBuffer.rewind()
        vBuffer.rewind()
        var offset = ySize
        for (row in 0 until chromaHeight) {
            for (col in 0 until chromaWidth) {
                val vu = row * vPlane.rowStride + col * vPlane.pixelStride
                val uu = row * uPlane.rowStride + col * uPlane.pixelStride
                if (vu >= vBuffer.limit() || uu >= uBuffer.limit()) return null
                out[offset++] = vBuffer.get(vu)
                out[offset++] = uBuffer.get(uu)
            }
        }
        return out
    }

    private fun copyPlaneRowWise(
        plane: ImageProxy.PlaneProxy,
        width: Int,
        height: Int,
        out: ByteArray,
        offset: Int,
        outPixelStride: Int
    ) {
        val buffer = plane.buffer.duplicate()
        buffer.rewind()
        val rowStride = plane.rowStride
        val pixelStride = plane.pixelStride
        var outIndex = offset
        for (row in 0 until height) {
            for (col in 0 until width) {
                out[outIndex] = buffer.get(row * rowStride + col * pixelStride)
                outIndex += outPixelStride
            }
        }
    }

    companion object {
        private const val TAG = "ImageAnalyzer"
        private const val TAG_DEBUG = "YOLO_DEBUG"
        private const val MODEL_SIZE = 640
        private const val YOLO_FEATURE_ROWS = 9
        private const val YOLO_ANCHORS = 8400
        private const val YOLO_CLASS_START_ROW = 4
        private const val YOLO_NUM_CLASSES = 5
        private const val CONF_THRESHOLD = 0.70f
        private const val CLASS_MARGIN_THRESHOLD = 0.14f
        private const val NMS_IOU_THRESHOLD = 0.30f
        private const val MAX_DETECTIONS = 6
        private const val MAX_BOXES_PER_CLASS = 1
        private const val MIN_BOX_SIZE_PX = 12f
        private const val MAX_BOX_AREA_RATIO = 0.65f
        private const val MIN_BOX_ASPECT_RATIO = 0.50f
        private const val MAX_BOX_ASPECT_RATIO = 2.00f
        private const val STABILITY_IOU_THRESHOLD = 0.25f
        private const val STABLE_FRAMES_REQUIRED = 2
        private const val INFERENCE_EVERY_N_FRAMES = 1L
        private const val ENABLE_TELEMETRY = false
        private const val ENABLE_DEBUG_LOGS = false
    }
}
