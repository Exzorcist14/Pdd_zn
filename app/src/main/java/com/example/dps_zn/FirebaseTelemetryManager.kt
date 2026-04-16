package com.example.dps_zn

import com.google.firebase.firestore.FieldValue
import com.google.firebase.firestore.FirebaseFirestore

/**
 * Edge-слой телеметрии: отправка событий распознавания в Firestore.
 */
object FirebaseTelemetryManager {

    private val firestore: FirebaseFirestore by lazy { FirebaseFirestore.getInstance() }

    fun sendSignData(name: String, conf: Double) {
        try {
            val payload = hashMapOf(
                "name" to name,
                "confidence" to conf,
                "ts" to FieldValue.serverTimestamp()
            )
            firestore.collection("detections").add(payload)
        } catch (_: Exception) {
            // Нет сети / неверный google-services.json — приложение не должно падать
        }
    }
}
