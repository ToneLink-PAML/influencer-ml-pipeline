const url = 'http://127.0.0.1:8000/recommend'

const btn = document.getElementById('s1')
const resp = document.getElementById('resp')

btn.addEventListener('click', handleSubmit)

async function handleSubmit(event) {
    event.preventDefault() // prevent page reload

    // Grab the values
    const name = document.getElementById('name').value
    const email = document.getElementById('email').value

    await sendData(name, email)
}


async function sendData(name, email) {  

    // Payload schema
    // TODO: Implement a proper schema
    const payload = {
        name: name,
        email: email,
    }

    // HTTP Request Responce
    const res = await fetch(url, {
        method: 'POST',
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(payload)
    })

    const data = await res.json()

    // Set the responce para
    resp.textContent = data['processed-data']['name']
    console.log('Server Replied:', data)
}