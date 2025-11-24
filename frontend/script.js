const url = 'http://127.0.0.1:8000/recommend'

document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('brand-form');
    const resultsContainer = document.getElementById('results-container');
    const initialMessage = document.getElementById('initial-message');

    /**
     * Renders the list of matched influencers into the container.
     */
    function renderResults(matches, brandData) {
        if (initialMessage) {
            initialMessage.classList.add('hidden');
        }

        resultsContainer.innerHTML = `
            <p class="text-black mb-6 font-semibold text-lg">
                Top 20 Matches for <span class="text-blue-900 font-bold">${brandData.brandName}</span>:
            </p>
        `;

        // Header row
        resultsContainer.innerHTML += `
            <div class="flex justify-between items-center px-5 py-3 border-b-2 border-black/20 font-bold text-sm text-subtle-text hidden sm:flex">
                <div class="flex-grow min-w-0">INFLUENCER / REGION</div>
                <div class="flex-shrink-0 flex space-x-8 text-right ml-auto">
                    <div class="w-24">FOLLOWERS</div>
                    <div class="w-28">SIMILARITY</div>
                    <div class="w-28">COMPATIBILITY</div>
                    <div class="w-20">MATCH</div>
                </div>
            </div>
        `;

        matches.forEach((influencer, index) => {
            const matchElement = document.createElement('div');
            matchElement.className =
                'influencer-card flex justify-between items-center p-5 content-block rounded-xl hover:translate-y-[-2px] hover:shadow-lg transition duration-200';
            matchElement.setAttribute('data-influencer-id', influencer.rank);

            // Left section: Influencer info
            const influencerProfile = `
                <div class="flex-grow min-w-0">
                    <p class="text-xl font-extrabold text-black truncate">
                        <span class="match-rank-prefix">${index + 1}.</span>
                        ${influencer.influencer_name}
                    </p>
                    <p class="text-sm text-subtle-text font-semibold ml-6">${influencer.region} (${influencer.platform})</p>
                </div>
            `;

            // Right section: Metrics
            const detailsGrid = `
                <div class="flex-shrink-0 flex space-x-8 text-right ml-auto">

                    <div class="w-24">
                        <p class="font-extrabold text-black text-lg">${(influencer.follower_count / 1000).toLocaleString(undefined, { maximumFractionDigits: 0 })}K</p>
                        <p class="text-xs text-subtle-text">Followers</p>
                    </div>

                    <div class="w-28">
                        <p class="font-bold text-black text-lg">${(influencer.similarity_score * 100).toFixed(0)}%</p>
                        <p class="text-xs text-subtle-text">Similarity</p>
                    </div>

                    <div class="w-28">
                        <p class="font-bold text-blue-900 text-lg">${(influencer.compatibility_score * 100).toFixed(0)}%</p>
                        <p class="text-xs text-subtle-text">Compatibility</p>
                    </div>

                    <div class="w-20">
                        <p class="font-extrabold text-green-800 text-xl">${influencer.final_match_score.toFixed(0)}</p>
                        <p class="text-xs text-subtle-text">Match</p>
                    </div>
                </div>
            `;

            matchElement.innerHTML = influencerProfile + detailsGrid;
            resultsContainer.appendChild(matchElement);
        });
    }


    // --- Form Submission Handler ---
    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const data = new FormData(form);
        const brandData = {
            brandName: data.get('brandName'),
            industry: data.get('industry'),
            audience: data.get('audience'),
            budget: parseInt(data.get('budget')) || 0,
            gender: data.get('gender'),
            region: data.get('region'),
            customerSegment: data.get('customerSegment'),
            description: data.get('description'),
            platform: data.get('platform'),
        };

        let responce = await sendData(brandData);

        window.currentBrandData = brandData;

        // Inject loading UI
        resultsContainer.innerHTML = `
            <div class="w-full mt-6">
                <p class="text-center text-lg font-semibold text-black mb-4">
                    Matching influencers… hold tight 👀
                </p>

                <div id="loading-bar-container" class="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
                    <div id="loading-bar" class="h-full bg-red-600 rounded-full" style="width: 0%;"></div>
                </div>
            </div>
        `;

        // Animate the bar for 5 seconds
        const bar = document.getElementById('loading-bar');
        let width = 0;

        const interval = setInterval(() => {
            width += 2; // 2% per 100ms = 100% in 5 sec
            bar.style.width = width + '%';

            if (width >= 100) clearInterval(interval);
        }, 100);

        // Wait 5 seconds
        await new Promise(resolve => setTimeout(resolve, 5000));

        // Show actual response
        renderResults(responce, brandData);
    });


    // Initial render on load (Simulate form submission)
    // form.dispatchEvent(new Event('submit'));
});

// Function to collect user input


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

    const parsed = JSON.parse(data);
    parsed.forEach(item => {  console.log(`${item.rank}. ${item.influencer_name} (${item.platform})`); });

    return parsed
  } catch (err) {
    console.error('Request failed:', err)
  }
}