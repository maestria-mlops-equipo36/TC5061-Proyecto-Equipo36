import random
from locust import HttpUser, task, between


class MiBiciAPIUser(HttpUser):
    """Simula usuarios concurrentes consultando predicciones de demanda diaria en MiBici."""

    wait_time = between(0.2, 1.0)

    @task(1)
    def health_check(self):
        with self.client.get("/health", name="/health", catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"Health check fallo con codigo {response.status_code}")

    @task(9)
    def predict_demand(self):
        base_demand = random.uniform(5.0, 65.0)
        dia_semana = random.randint(0, 6)
        es_fin = 1 if dia_semana >= 5 else 0

        payload = {
            "estacion_id": random.randint(1, 348),
            "dia_semana": dia_semana,
            "mes": random.randint(1, 12),
            "es_fin_de_semana": es_fin,
            "viajes_dia_anterior": round(max(0.0, base_demand + random.uniform(-6.0, 6.0)), 2),
            "viajes_hace_7_dias": round(max(0.0, base_demand + random.uniform(-8.0, 8.0)), 2),
            "promedio_7_dias": round(base_demand, 2),
            "promedio_14_dias": round(max(0.0, base_demand + random.uniform(-4.0, 4.0)), 2),
            "viajes_hace_14_dias": round(max(0.0, base_demand + random.uniform(-9.0, 9.0)), 2),
            "viajes_hace_28_dias": round(max(0.0, base_demand + random.uniform(-10.0, 10.0)), 2),
            "promedio_28_dias": round(max(0.0, base_demand + random.uniform(-5.0, 5.0)), 2),
            "promedio_mismo_dia_4_semanas": round(max(0.0, base_demand + random.uniform(-4.5, 4.5)), 2),
        }

        with self.client.post(
            "/predict",
            json=payload,
            name="/predict",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}: {response.text}")
            else:
                data = response.json()
                if "demanda_estimada" not in data or data["demanda_estimada"] < 0:
                    response.failure(f"Respuesta invalida: {data}")
