package com.example.dps_zn

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import kotlin.math.min

/**
 * Наложение поверх превью камеры: рамки и подписи знаков в координатах исходного кадра анализа.
 */
class GraphicOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    data class OverlayDetection(
        val label: String,
        val confidence: Float,
        /** Координаты в системе исходного изображения (после поворота), пиксели */
        val rectImage: RectF
    )

    private val lock = Any()
    private var imageWidth: Int = 0
    private var imageHeight: Int = 0
    private var detections: List<OverlayDetection> = emptyList()
    private var mirrorX: Boolean = false

    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 4f
        color = Color.argb(220, 0, 200, 100)
    }
    private val textBgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = Color.argb(200, 0, 0, 0)
    }
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 36f
    }

    fun setSourceImageSize(width: Int, height: Int) {
        synchronized(lock) {
            imageWidth = width
            imageHeight = height
        }
        postInvalidateOnAnimation()
    }

    fun setDetections(items: List<OverlayDetection>) {
        synchronized(lock) {
            detections = items
        }
        postInvalidateOnAnimation()
    }

    fun setMirrorX(value: Boolean) {
        synchronized(lock) {
            mirrorX = value
        }
        postInvalidateOnAnimation()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val iw: Int
        val ih: Int
        val list: List<OverlayDetection>
        val needMirror: Boolean
        synchronized(lock) {
            iw = imageWidth
            ih = imageHeight
            list = detections
            needMirror = mirrorX
        }
        if (iw <= 0 || ih <= 0 || width == 0 || height == 0) return

        val scaleX = width.toFloat() / iw
        val scaleY = height.toFloat() / ih
        val scale = min(scaleX, scaleY)
        val dx = (width - iw * scale) / 2f
        val dy = (height - ih * scale) / 2f

        for (d in list) {
            var left = d.rectImage.left * scale + dx
            var right = d.rectImage.right * scale + dx
            val top = d.rectImage.top * scale + dy
            val bottom = d.rectImage.bottom * scale + dy

            if (needMirror) {
                val mirroredLeft = width - right
                val mirroredRight = width - left
                left = mirroredLeft
                right = mirroredRight
            }

            val r = RectF(
                left.coerceIn(0f, width.toFloat()),
                top.coerceIn(0f, height.toFloat()),
                right.coerceIn(0f, width.toFloat()),
                bottom.coerceIn(0f, height.toFloat())
            )
            if (r.width() <= 1f || r.height() <= 1f) continue
            canvas.drawRect(r, boxPaint)
            val label = d.label
            val pad = 8f
            val tw = textPaint.measureText(label)
            val th = textPaint.fontMetrics.let { it.descent - it.ascent }
            val bg = RectF(r.left, r.top - th - pad * 2, r.left + tw + pad * 2, r.top)
            canvas.save()
            canvas.drawRect(bg, textBgPaint)
            canvas.drawText(label, bg.left + pad, bg.bottom - pad - textPaint.fontMetrics.descent, textPaint)
            canvas.restore()
        }
    }
}
