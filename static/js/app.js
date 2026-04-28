/* Bangalore Traffic AI - Main JavaScript */

// Mobile Nav Toggle
document.addEventListener("DOMContentLoaded", () => {
    const toggle = document.querySelector(".nav-toggle");
    const links = document.querySelector(".nav-links");
    if (toggle && links) {
        toggle.addEventListener("click", () => links.classList.toggle("open"));
    }
    // Set active nav link
    const path = window.location.pathname;
    document.querySelectorAll(".nav-links a").forEach(a => {
        if (a.getAttribute("href") === path) a.classList.add("active");
    });
});

// Loader
function showLoader() {
    const el = document.getElementById("loader");
    if (el) el.classList.add("active");
}
function hideLoader() {
    const el = document.getElementById("loader");
    if (el) el.classList.remove("active");
}

// Prediction Form
async function submitPrediction(e) {
    e.preventDefault();
    showLoader();
    const form = document.getElementById("predict-form");
    const data = Object.fromEntries(new FormData(form));
    try {
        const res = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data)
        });
        const json = await res.json();
        if (json.status === "success") {
            displayResults(json.data);
        } else {
            alert("Prediction error: " + json.message);
        }
    } catch (err) {
        alert("Network error: " + err.message);
    }
    hideLoader();
}

function displayResults(data) {
    const panel = document.getElementById("results-panel");
    if (!panel) return;
    panel.classList.add("active");

    // Congestion badge
    const level = data.congestion.congestion_level.toLowerCase();
    const badge = document.getElementById("congestion-badge");
    if (badge) {
        badge.className = "congestion-badge " + level;
        badge.textContent = data.congestion.congestion_level;
    }
    const conf = document.getElementById("confidence");
    if (conf) conf.textContent = data.congestion.confidence + "%";

    // ETA
    setText("eta-value", data.eta.eta_minutes + " min");
    setText("distance-value", data.eta.distance_km + " km");
    setText("speed-value", data.eta.avg_speed + " km/h");
    setText("traffic-index-value", data.congestion.traffic_index);

    // Best departure
    if (data.departure) {
        setText("best-dep-value", formatHour(data.departure.best_departure_hour));
        setText("best-dep-traffic", "Traffic Index: " + data.departure.best_traffic_index);
    }

    // Routes
    const routesCont = document.getElementById("routes-container");
    if (routesCont && data.routes && data.routes.routes) {
        routesCont.innerHTML = data.routes.routes.map(r => `
            <div class="route-card ${r.recommended ? 'recommended' : ''}">
                <div class="route-info">
                    <h4>${r.route_type} ${r.recommended ? '⭐ Recommended' : ''}</h4>
                    <div class="route-meta">
                        <span>🕐 ${r.eta_minutes} min</span>
                        <span>📏 ${r.distance_km} km</span>
                        <span>⚠️ Risk: ${r.risk_score}</span>
                    </div>
                </div>
                <span class="congestion-badge ${r.congestion.toLowerCase()}">${r.congestion}</span>
            </div>
        `).join("");
    }

    // Forecast table
    const fcBody = document.getElementById("forecast-body");
    if (fcBody && data.forecast && data.forecast.forecasts) {
        fcBody.innerHTML = data.forecast.forecasts.map(f => `
            <tr>
                <td>${f.time_label}</td>
                <td>${f.traffic_index}</td>
                <td><span class="congestion-badge ${f.congestion_level.toLowerCase()}" style="font-size:0.75rem;padding:3px 10px;">${f.congestion_level}</span></td>
            </tr>
        `).join("");
    }

    // Mini map
    if (data.source_coords && data.dest_coords) {
        initResultMap(data.source_coords, data.dest_coords, data.source, data.destination, level);
    }

    panel.scrollIntoView({ behavior: "smooth", block: "start" });
}

function setText(id, val) {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
}

function formatHour(h) {
    if (h === 0) return "12:00 AM";
    if (h < 12) return h + ":00 AM";
    if (h === 12) return "12:00 PM";
    return (h - 12) + ":00 PM";
}

// Result mini-map
let resultMap = null;
function initResultMap(src, dst, srcName, dstName, level) {
    const el = document.getElementById("result-map");
    if (!el) return;
    if (resultMap) resultMap.remove();
    const midLat = (src[0] + dst[0]) / 2;
    const midLng = (src[1] + dst[1]) / 2;
    resultMap = L.map("result-map").setView([midLat, midLng], 12);
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        attribution: '&copy; OSM'
    }).addTo(resultMap);

    const colors = { low: "#10b981", medium: "#f59e0b", high: "#ef4444", severe: "#ec4899" };
    const color = colors[level] || "#3b82f6";

    L.marker(src).addTo(resultMap).bindPopup(`<b>📍 ${srcName}</b><br>Source`);
    L.marker(dst).addTo(resultMap).bindPopup(`<b>📍 ${dstName}</b><br>Destination`);
    L.polyline([src, dst], { color: color, weight: 4, dashArray: "8,8" }).addTo(resultMap);
    L.circle(src, { radius: 800, color: color, fillOpacity: 0.15 }).addTo(resultMap);
    L.circle(dst, { radius: 800, color: color, fillOpacity: 0.15 }).addTo(resultMap);
    setTimeout(() => resultMap.invalidateSize(), 200);
}

// Dashboard Charts & Auto-Refresh
const chartInstances = {};
let dashboardRefreshInterval = null;

async function loadDashboard() {
    const isFirstLoad = Object.keys(chartInstances).length === 0;
    if (isFirstLoad) showLoader();
    
    try {
        const res = await fetch("/api/analytics");
        const json = await res.json();
        if (json.status === "success") {
            renderDashboard(json.data);
            updateLastUpdated();
        }
    } catch (err) {
        console.error("Dashboard error:", err);
    }
    
    if (isFirstLoad) hideLoader();

    // Set up auto-refresh if not already set
    if (!dashboardRefreshInterval) {
        dashboardRefreshInterval = setInterval(loadDashboard, 30000); // Refresh every 30s
    }
}

function updateLastUpdated() {
    const el = document.getElementById("last-updated");
    if (el) {
        const now = new Date();
        el.textContent = `Last updated: ${now.toLocaleTimeString()}`;
    }
}

const chartColors = {
    blue: "rgba(59,130,246,0.8)", cyan: "rgba(6,182,212,0.8)",
    purple: "rgba(139,92,246,0.8)", green: "rgba(16,185,129,0.8)",
    orange: "rgba(245,158,11,0.8)", red: "rgba(239,68,68,0.8)", pink: "rgba(236,72,153,0.8)"
};
const chartDefaults = {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { labels: { color: "#94a3b8", font: { family: "Inter" } } } },
    scales: {
        x: { ticks: { color: "#64748b" }, grid: { color: "rgba(255,255,255,0.04)" } },
        y: { ticks: { color: "#64748b" }, grid: { color: "rgba(255,255,255,0.04)" } }
    }
};

function renderDashboard(data) {
    if (!data) return;
    
    // Stats
    if (data.stats) {
        setText("stat-records", (data.stats.total_records || 0).toLocaleString());
        setText("stat-avg-traffic", data.stats.avg_traffic_index || 0);
        setText("stat-avg-travel", data.stats.avg_travel_time || 0);
        setText("stat-avg-speed", data.stats.avg_speed || 0);
        setText("stat-severe", (data.stats.severe_pct || 0) + "%");
    }
    
    // Charts - Reusing instances if they exist
    if (data.hourly_traffic) renderChart("hourlyChart", "line", data.hourly_traffic, "Hourly Traffic Pattern");
    if (data.weather_impact) renderChart("weatherChart", "bar", data.weather_impact, "Weather Impact", true);
    if (data.weekday_traffic) renderWeekdayChart(data.weekday_traffic);
    if (data.congestion_distribution) renderCongestionPie(data.congestion_distribution);
    if (data.top_congested_routes) renderTopRoutes(data.top_congested_routes);
    if (data.area_traffic) renderChart("areaChart", "bar", data.area_traffic, "Area-wise Traffic", true);
}

function renderChart(canvasId, type, d, label, horizontal = false) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    const labels = Object.keys(d);
    const vals = Object.values(d);

    if (chartInstances[canvasId]) {
        // Update existing chart
        chartInstances[canvasId].data.labels = labels;
        chartInstances[canvasId].data.datasets[0].data = vals;
        chartInstances[canvasId].update();
    } else {
        // Create new chart
        chartInstances[canvasId] = new Chart(ctx, {
            type: type,
            data: {
                labels: labels,
                datasets: [{
                    label: "Avg Traffic Index",
                    data: vals,
                    backgroundColor: type === "line" ? "rgba(59,130,246,0.1)" : Object.values(chartColors),
                    borderColor: chartColors.blue,
                    fill: type === "line",
                    tension: 0.4
                }]
            },
            options: { ...chartDefaults, indexAxis: horizontal ? "y" : "x" }
        });
    }
}

function renderWeekdayChart(d) {
    const canvasId = "weekdayChart";
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;
    
    const days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
    const vals = [0,1,2,3,4,5,6].map(i => d[String(i)] || 0);

    if (chartInstances[canvasId]) {
        chartInstances[canvasId].data.datasets[0].data = vals;
        chartInstances[canvasId].update();
    } else {
        chartInstances[canvasId] = new Chart(ctx, {
            type: "bar",
            data: {
                labels: days,
                datasets: [{
                    label: "Avg Traffic Index",
                    data: vals,
                    backgroundColor: vals.map((v, i) => i >= 5 ? chartColors.green : chartColors.purple)
                }]
            },
            options: chartDefaults
        });
    }
}

function renderCongestionPie(d) {
    const canvasId = "congestionPie";
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    if (chartInstances[canvasId]) {
        chartInstances[canvasId].data.labels = Object.keys(d);
        chartInstances[canvasId].data.datasets[0].data = Object.values(d);
        chartInstances[canvasId].update();
    } else {
        chartInstances[canvasId] = new Chart(ctx, {
            type: "doughnut",
            data: {
                labels: Object.keys(d),
                datasets: [{
                    data: Object.values(d),
                    backgroundColor: [chartColors.green, chartColors.orange, chartColors.red, chartColors.pink]
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { position: "bottom", labels: { color: "#94a3b8" } } }
            }
        });
    }
}

function renderTopRoutes(d) {
    const cont = document.getElementById("top-routes-list");
    if (!cont) return;
    cont.innerHTML = d.slice(0, 8).map((r, i) => `
        <div class="route-item" style="display:flex;justify-content:space-between;align-items:center;padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.05);">
            <span style="font-size:0.85rem;">${i + 1}. ${r.source} → ${r.destination}</span>
            <span style="font-size:0.85rem;color:${r.avg_traffic > 60 ? '#ef4444' : '#f59e0b'};font-weight:600;">${r.avg_traffic}</span>
        </div>
    `).join("");
}

// Traffic Map
let trafficMap = null;
function initTrafficMap(locationsData) {
    const el = document.getElementById("traffic-map");
    if (!el) return;
    if (trafficMap) trafficMap.remove();
    trafficMap = L.map("traffic-map").setView([12.9716, 77.5946], 12);
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        attribution: '&copy; <a href="https://osm.org/copyright">OpenStreetMap</a>',
        maxZoom: 18
    }).addTo(trafficMap);
    
    // Fix for Leaflet marker icons not showing up
    delete L.Icon.Default.prototype._getIconUrl;
    L.Icon.Default.mergeOptions({
        iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
        iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
        shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
    });

    if (locationsData) {
        const locs = typeof locationsData === "string" ? JSON.parse(locationsData) : locationsData;
        Object.entries(locs).forEach(([name, coords]) => {
            const congestion = Math.random();
            let color = "#10b981";
            if (congestion > 0.75) color = "#ec4899";
            else if (congestion > 0.5) color = "#ef4444";
            else if (congestion > 0.25) color = "#f59e0b";
            L.circle(coords, { radius: 600, color: color, fillColor: color, fillOpacity: 0.25, weight: 2 })
                .addTo(trafficMap).bindPopup(`<b>${name}</b><br>Traffic Zone`);
            L.marker(coords).addTo(trafficMap).bindPopup(`<b>${name}</b>`);
        });
    }
    // Ensure map is correctly sized
    setTimeout(() => trafficMap.invalidateSize(), 300);
}

// Animate counter
function animateValue(el, start, end, duration) {
    const range = end - start;
    const startTime = performance.now();
    function update(now) {
        const elapsed = now - startTime;
        const progress = Math.min(elapsed / duration, 1);
        el.textContent = Math.floor(start + range * progress).toLocaleString();
        if (progress < 1) requestAnimationFrame(update);
    }
    requestAnimationFrame(update);
}

// Init counters on home page
function initCounters() {
    document.querySelectorAll("[data-count]").forEach(el => {
        const target = parseInt(el.dataset.count);
        const observer = new IntersectionObserver((entries) => {
            if (entries[0].isIntersecting) {
                animateValue(el, 0, target, 2000);
                observer.disconnect();
            }
        });
        observer.observe(el);
    });
}
document.addEventListener("DOMContentLoaded", initCounters);
