document.getElementById('prediction-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const form = e.target;
    const btn = form.querySelector('.btn-predict');
    const btnText = document.getElementById('btn-text');
    const loader = document.getElementById('loader');
    const placeholder = document.getElementById('result-placeholder');
    const display = document.getElementById('result-display');
    
    // UI Loading State
    btn.disabled = true;
    btnText.style.display = 'none';
    loader.style.display = 'block';
    
    const formData = new FormData(form);
    const payload = {
        area_name: formData.get('area_name'),
        hour: parseInt(formData.get('hour')),
        day_of_week: formData.get('day_of_week'),
        holiday: parseInt(formData.get('holiday')),
        weather: formData.get('weather'),
        rainfall: parseFloat(formData.get('rainfall')),
        road_type: formData.get('road_type'),
        event_nearby: parseInt(formData.get('event_nearby')),
        accident_reported: parseInt(formData.get('accident_reported')),
        traffic_volume: parseInt(formData.get('traffic_volume')),
        route_distance: parseFloat(formData.get('route_distance'))
    };

    const startTime = performance.now();

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const result = await response.json();
        const latency = Math.round(performance.now() - startTime);

        if (response.ok) {
            placeholder.style.display = 'none';
            display.style.display = 'block';
            
            // Update Results
            const level = result.congestion_level;
            const resLevel = document.getElementById('res-level');
            const resBadge = document.getElementById('res-badge');
            
            resLevel.textContent = level;
            resBadge.textContent = `${(result.confidence * 100).toFixed(0)}% Certainty`;
            
            // Update Badge Color
            resBadge.className = 'badge';
            resBadge.classList.add(`badge-${level.toLowerCase()}`);
            
            document.getElementById('res-time').textContent = `${result.estimated_travel_time_minutes} Minutes`;
            document.getElementById('res-suggestion').textContent = result.route_suggestion;
            document.getElementById('res-conf').textContent = `${(result.confidence * 100).toFixed(0)}%`;
            document.getElementById('res-lat').textContent = `${latency}ms`;
        } else {
            alert(`Error: ${result.detail || 'Internal Server Error'}`);
        }
    } catch (error) {
        console.error('Fetch error:', error);
        alert('Could not connect to the Prediction API. Is the server running?');
    } finally {
        btn.disabled = false;
        btnText.style.display = 'inline';
        loader.style.display = 'none';
    }
});
