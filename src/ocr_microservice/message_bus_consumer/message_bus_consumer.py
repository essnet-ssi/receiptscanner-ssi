import os
import pika
import json
import signal
import concurrent.futures

from ocr_microservice.ocr_pipeline.config.default import Config, Models, Pipeline
from ocr_microservice.ocr_pipeline.injector import Injector
from ocr_microservice.ocr_pipeline.ocr_pipeline import process


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


def stop_consuming(channel):
    def handler(signal, frame):
        print("Signal received, shutting down...")
        channel.stop_consuming()

    return handler


def run_ocr_with_timeout(image, filename, injector, timeout_seconds=300):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(process, image, filename, injector)
        return future.result(timeout=timeout_seconds)


def on_message(channel, method_frame, header_frame, body):
    try:
        message = json.loads(body)
        result = run_ocr_with_timeout(message["image"], message["filename"], injector)
        channel.basic_publish(
            exchange="",
            routing_key="receipt_results_queue",
            body=json.dumps({"id": message["id"], "result": result}),
        )
        channel.basic_ack(delivery_tag=method_frame.delivery_tag)
    except Exception as e:
        # TODO: Log the exception to persistent storage
        print(f"Error processing message: {e}")
        # TODO: Implement max retries
        channel.basic_nack(delivery_tag=method_frame.delivery_tag, requeue=True)


if __name__ == "__main__":
    channel = create_channel()
    signal.signal(signal.SIGTERM, lambda signum, frame: stop_consuming(channel))
    signal.signal(signal.SIGINT, lambda signum, frame: stop_consuming(channel))
    injector = Injector(Config(), Pipeline(), Models())

    channel.basic_consume(
        queue="receipt_requests_queue", on_message_callback=on_message
    )
    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        channel.stop_consuming()
