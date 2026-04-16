package com.example.dps_zn

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.Toast
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var overlay: GraphicOverlayView

    /** Только кадры ImageAnalysis — не смешивать с блокирующим getInstance(). */
    private val analysisExecutor = Executors.newSingleThreadExecutor { r ->
        Thread(r, "dps-image-analysis").apply { isDaemon = true }
    }

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted: Boolean ->
        if (granted) startCamera()
        else Toast.makeText(this, "Нужен доступ к камере", Toast.LENGTH_LONG).show()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(bars.left, bars.top, bars.right, bars.bottom)
            insets
        }

        previewView = findViewById(R.id.preview_view)
        overlay = findViewById(R.id.graphic_overlay)
        previewView.implementationMode = PreviewView.ImplementationMode.COMPATIBLE
        previewView.scaleType = PreviewView.ScaleType.FIT_CENTER

        when {
            ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
                PackageManager.PERMISSION_GRANTED -> startCamera()
            else -> permissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        analysisExecutor.shutdown()
    }

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener(
            {
                try {
                    val provider = future.get()
                    bindCameraUseCases(provider)
                } catch (e: Exception) {
                    Toast.makeText(this, "Камера: ${e.message}", Toast.LENGTH_LONG).show()
                }
            },
            ContextCompat.getMainExecutor(this)
        )
    }

    private fun bindCameraUseCases(provider: ProcessCameraProvider) {
        val targetRotation = previewView.display?.rotation ?: android.view.Surface.ROTATION_0
        val preview = Preview.Builder()
            .setTargetRotation(targetRotation)
            .build().also {
            it.surfaceProvider = previewView.surfaceProvider
        }

        val analysis = ImageAnalysis.Builder()
            .setTargetRotation(targetRotation)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()

        val analyzer = ImageAnalyzer(applicationContext) { iw, ih, dets ->
            overlay.post {
                overlay.setSourceImageSize(iw, ih)
                overlay.setDetections(dets)
            }
        }
        analysis.setAnalyzer(analysisExecutor, analyzer)

        provider.unbindAll()
        if (tryBind(provider, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)) {
            overlay.setMirrorX(false)
            return
        }

        provider.unbindAll()
        if (tryBind(provider, CameraSelector.DEFAULT_FRONT_CAMERA, preview, analysis)) {
            overlay.setMirrorX(true)
            return
        }

        Toast.makeText(
            this,
            "Камера (задняя/фронтальная): не удалось привязать use cases",
            Toast.LENGTH_LONG
        ).show()
    }

    private fun tryBind(
        provider: ProcessCameraProvider,
        selector: CameraSelector,
        preview: Preview,
        analysis: ImageAnalysis
    ): Boolean {
        return try {
            provider.bindToLifecycle(this, selector, preview, analysis)
            true
        } catch (_: Exception) {
            false
        }
    }
}
