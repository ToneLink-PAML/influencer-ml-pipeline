from fastapi import APIRouter

from schema.payload import Payload

router = APIRouter()

# POST is used to send data and fetch the processed.
# Here we are writing the logic when a user gives the data.
# We process the data and send it back.
@router.post('/recommend')
def recommend_influencer(payload: Payload):
    print([payload])
    payload.name += 'From FAPI'
    return {'processed-data': payload}

# This is called when a user/browser trys to fetch data.
@router.get('/influencer/{id}')
def get_influencer_id(id: int):
    return {'data': 'no-data'}
