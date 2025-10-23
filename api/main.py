from fastapi import FastAPI

app = FastAPI(
    title = "Influencer ML Model Pipeline",
    description= "All purpose pipline which aggregates all the ml models and generates a comprehensible output.",
    version= "1.0.0"
)

# POST is used to send data and fetch the processed.
# Here we are writing the logic when a user gives the data.
# We process the data and send it back.
@app.post('/recommend')
def recommend_influencer(data):
    return data

# This is called when a user/browser trys to fetch data.
@app.get('/influencer/{id}')
def get_influencer_id(id: int):
    return {'data': 'no-data'}