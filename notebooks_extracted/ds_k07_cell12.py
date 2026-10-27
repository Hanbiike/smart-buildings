from kaggle_evaluation import smart_buildings_inference_server

class SubmissionPolicy(TunedDemandSatPolicy, smart_buildings_inference_server.Policy):
    KP = 0.07
    KI = 0.0035
    HOLD_STEPS = 288
    NIGHT_COOL_CEILING = True
    FORCED_PRESSURE = -1.0
    NIGHT_HEAT_DROP_K = 2.0
    PRESTART_H = 2
