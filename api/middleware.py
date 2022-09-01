import time
import settings
import json
import redis
import uuid

# Connecting to Redis and assign to variable `db``
# Making use of settings.py module to get Redis settings like host, port, etc.
db = redis.Redis(
    host=settings.REDIS_IP, 
    port=settings.REDIS_PORT, 
    db=settings.REDIS_DB_ID
)


def model_predict(form_dict):
    """
    Receives a dictionary and queues the job into Redis.
    Will loop until getting the answer from our ML service.

    Parameters
    ----------
    form_dict : dict
        Name for the dictionary uploaded by the user via form.

    Returns
    -------
    prediction, prediction_proba : tuple(int, float)
        Model predicted class as a integer and the corresponding probability
        prediction of target 1 as a number.
    """
    # Assigning an unique ID for this job and add it to the queue.
    # We need to assing this ID because we must be able to keep track
    # of this particular job across all the services
    job_id = str(uuid.uuid4())

    # Creating a dict with the job data we will send through Redis having the
    # following shape:
    # {
    #    "id": str,
    #    "form_dict": dict,
    # }
    job_data = {
        'id': job_id,
        'form_dict': form_dict
    }

    # Sending the job to the model service using Redis
    # Note: Using Redis `rpush()` function should be enough to accomplish this.
    # TODO
    job_data_redis = json.dumps(job_data)
    db.rpush(settings.REDIS_QUEUE, job_data_redis)

    # Looping until we received the response from our ML model
    while True:
        # Attempt to get model predictions using job_id
        if db.get(job_id):
            output = json.loads(db.get(job_id))
            # Deleting the job from Redis after we get the results!
            db.delete(job_id)
            # Ending the loop
            break

        # Sleep some time waiting for model results
        time.sleep(settings.API_SLEEP)

    return output['prediction'], output['prediction_proba']
