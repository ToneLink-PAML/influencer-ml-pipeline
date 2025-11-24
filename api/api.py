from fastapi import APIRouter

from schema.payload import Payload
from pipeline.pip import load_models
from input_embedder import BrandEmbeddingGenerator

router = APIRouter()

# POST is used to send data and fetch the processed.
# Here we are writing the logic when a user gives the data.
# We process the data and send it back.
@router.post('/recommend')
async def recommend_influencer(payload: Payload):
    print(payload)
    emb = BrandEmbeddingGenerator(payload=payload).create_embeddings()
    # comment the above line if it dosent work.
    result = load_models()
    return result

# This is called when a user/browser trys to fetch data.
@router.get('/influencer/{id}')
def get_influencer_id(id: int):
    return {'data': 'no-data'}
