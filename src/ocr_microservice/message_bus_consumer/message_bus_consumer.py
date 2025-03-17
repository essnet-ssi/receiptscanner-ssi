
import os
import gc
import pika
import json
import signal
import concurrent.futures
from PIL import Image

from ocr_microservice.ocr_pipeline.config.default import Cache, Config, Models, Pipeline
from ocr_microservice.ocr_pipeline.injector import Injector
from ocr_microservice.ocr_pipeline.ocr_pipeline import process
from ocr_microservice.ocr_pipeline.pipeline_image import PipelineImage
from ocr_microservice.ocr_pipeline.helpers.logger import log

# os.environ["IMAGE_FOLDER"] = "/Users/tom/work/receipts" for debugging locally

def create_channel():
    rabbitmq_host = os.getenv("RABBITMQ_HOST", "localhost")
    connection_params = pika.ConnectionParameters(
        host=rabbitmq_host, heartbeat=600, blocked_connection_timeout=300
    )
    connection = pika.BlockingConnection(connection_params)
    channel = connection.channel()
    channel.queue_declare(queue="receipt_requests_queue")
    channel.queue_declare(queue="receipt_results_queue")
    return channel

def stop_consuming(channel, connection):
    def handler(signal, frame):
        log("Signal received, shutting down...")
        channel.stop_consuming()
        connection.close()
    return handler

def run_ocr_with_timeout(pipeline_image, injector, timeout_seconds=300):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(process, pipeline_image, injector)
        return future.result(timeout=timeout_seconds)


def on_message(channel, method_frame, header_frame, body):
    log("Message received")
    injector = Injector(Config(), Pipeline(), Models(), Cache()) 
    image_filename_from_message = body.decode("utf-8").strip()
    log(image_filename_from_message)

    if not image_filename_from_message:
        log("Received an empty message body.")
        channel.basic_ack(delivery_tag=method_frame.delivery_tag)
        return

    image_folder = os.getenv("IMAGE_FOLDER", "/mnt/images")
    image_path = os.path.join(image_folder, image_filename_from_message)

    with Image.open(image_path) as image:
        pipeline_image = PipelineImage(image=image, filename=image_filename_from_message)

        try:
            result = run_ocr_with_timeout(pipeline_image, injector)
            if result is not None:
                channel.basic_publish(
                    exchange="",
                    routing_key="receipt_results_queue",
                    body=json.dumps({"id": image_filename_from_message, "result": result}),
                )
                channel.basic_ack(delivery_tag=method_frame.delivery_tag)
            del pipeline_image, result
            gc.collect()
        except concurrent.futures.TimeoutError:
            log("Processing timed out")
            channel.basic_nack(delivery_tag=method_frame.delivery_tag, requeue=True)
        except Exception as e:
            log(f"Error processing message: {e}")
            channel.basic_nack(delivery_tag=method_frame.delivery_tag, requeue=True)

if __name__ == "__main__":
    log("Starting consumer")
    channel = create_channel()
    signal.signal(signal.SIGTERM, lambda signum, frame: stop_consuming(channel))
    signal.signal(signal.SIGINT, lambda signum, frame: stop_consuming(channel))
    channel.basic_consume(queue="receipt_requests_queue", on_message_callback=on_message)
    log("Consumer is listening for messages...")
    try:
        channel.start_consuming()
    except Exception as e:
        channel.stop_consuming()
