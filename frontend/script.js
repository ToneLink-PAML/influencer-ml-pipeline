const url = 'http://127.0.0.1:8000/recommend'

const btn = document.getElementById('s1')
const resp = document.getElementById('resp')

// Function to collect user input
function getBrandInput() {
  return {
    brand_name: document.getElementById('brand_name').value,
    campaign_name: document.getElementById('campaign_name').value,
    description: document.getElementById('description').value,
    target_region: document.getElementById('target_region').value,
    target_age_group: document.getElementById('target_age_group').value,
    target_gender: document.getElementById('target_gender').value,
    keywords: document.getElementById('keywords').value
      .split(',')
      .map(k => k.trim())
      .filter(k => k.length > 0)  // Only keep non-empty strings
  };
}

btn.addEventListener('click', handleSubmit);

async function handleSubmit(event) {
  event.preventDefault()
  const brandInput = getBrandInput()
  await sendData(brandInput)
}

async function sendData(payload) {
  try {
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    })

    if (!res.ok) throw new Error(`HTTP error: ${res.status}`)

    const data = await res.json()
    console.log('Server replied:', data)

    // Display the response (edit as per your API response)
    resp.textContent = JSON.stringify(data, null, 2)
  } catch (err) {
    console.error('Request failed:', err)
  }
}
