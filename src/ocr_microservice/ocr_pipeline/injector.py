from ocr_microservice.ocr_pipeline.config.default import Config, Pipeline, Models, Cache


class Injector:
    def __init__(self, config: Config, pipeline: Pipeline, models: Models, cache: Cache):
        self.config = config
        self.pipeline = pipeline
        self.models = models
        self.cache = cache

