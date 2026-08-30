// ParkVision AI - Frontend Controller & Telemetry Poller

let chart = null;
const maxDataPoints = 20;
const chartLabels = [];
const occupancyData = [];

// Initialize Real-time Chart.js
function initChart() {
    const ctx = document.getElementById('occupancyChart').getContext('2d');
    
    // Gradient fill for chart
    const gradient = ctx.createLinearGradient(0, 0, 0, 160);
    gradient.addColorStop(0, 'rgba(6, 182, 212, 0.4)');
    gradient.addColorStop(1, 'rgba(6, 182, 212, 0.0)');

    chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: chartLabels,
            datasets: [{
                label: 'Occupancy %',
                data: occupancyData,
                borderColor: '#06b6d4',
                backgroundColor: gradient,
                borderWidth: 2,
                tension: 0.35,
                fill: true,
                pointBackgroundColor: '#06b6d4',
                pointRadius: 2,
                pointHoverRadius: 5
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            scales: {
                x: {
                    display: false
                },
                y: {
                    min: 0,
                    max: 100,
                    grid: {
                        color: 'rgba(51, 65, 85, 0.3)'
                    },
                    ticks: {
                        color: '#94a3b8',
                        font: { size: 10 },
                        callback: function(value) { return value + '%'; }
                    }
                }
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    backgroundColor: 'rgba(15, 23, 42, 0.9)',
                    titleColor: '#e2e8f0',
                    bodyColor: '#06b6d4',
                    borderColor: 'rgba(51, 65, 85, 0.5)',
                    borderWidth: 1
                }
            }
        }
    });
}

// Poll API for metrics
async function updateMetrics() {
    try {
        const res = await fetch('/api/space_count');
        if (!res.ok) return;

        const data = await res.json();
        if (data.status !== 'success') return;

        // 1. Update KPI Cards
        document.getElementById('card-total').innerText = data.total_spaces || 0;
        document.getElementById('card-free').innerText = data.free_spaces || 0;
        document.getElementById('card-occupied').innerText = data.occupied_spaces || 0;
        document.getElementById('card-rate').innerText = `${data.occupancy_rate || 0}%`;
        document.getElementById('progress-bar').style.width = `${data.occupancy_rate || 0}%`;
        document.getElementById('header-fps').innerText = `${data.fps || 0} FPS`;
        document.getElementById('stream-timestamp').innerText = data.timestamp || new Date().toLocaleTimeString();

        // 2. Update Slots Status Grid
        const container = document.getElementById('slots-container');
        document.getElementById('slots-count-badge').innerText = `${data.spaces.length} Spots`;

        if (data.spaces && data.spaces.length > 0) {
            let html = '';
            data.spaces.forEach(slot => {
                const isOcc = slot.is_occupied;
                const badgeBg = isOcc ? 'bg-rose-500/10 text-rose-400 border-rose-500/20' : 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20';
                const icon = isOcc ? 'fa-solid fa-car-side text-rose-400' : 'fa-regular fa-circle-check text-emerald-400';
                
                html += `
                    <div class="flex items-center justify-between p-2 rounded-xl bg-slate-800/40 border border-slate-800 hover:border-slate-700 transition">
                        <div class="flex items-center space-x-2.5">
                            <span class="w-6 h-6 rounded-lg bg-slate-800 flex items-center justify-center text-xs font-mono font-bold text-slate-300">
                                ${slot.id < 10 ? '0' + slot.id : slot.id}
                            </span>
                            <span class="text-xs font-medium text-slate-300">Slot P${slot.id < 10 ? '0' + slot.id : slot.id}</span>
                        </div>
                        <div class="flex items-center space-x-2">
                            <span class="text-[10px] font-mono text-slate-400">${slot.confidence}%</span>
                            <span class="px-2 py-0.5 rounded-md text-[10px] font-semibold border ${badgeBg} flex items-center space-x-1">
                                <i class="${icon} text-[10px]"></i>
                                <span>${slot.status}</span>
                            </span>
                        </div>
                    </div>
                `;
            });
            container.innerHTML = html;
        }

        // 3. Update Chart
        if (chart) {
            const now = new Date().toLocaleTimeString();
            chartLabels.push(now);
            occupancyData.push(data.occupancy_rate);

            if (chartLabels.length > maxDataPoints) {
                chartLabels.shift();
                occupancyData.shift();
            }
            chart.update();
        }

    } catch (err) {
        console.error('Error fetching parking metrics:', err);
    }
}

// Switch video source
async function changeSource(source) {
    try {
        const res = await fetch('/api/switch_source', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ source: source })
        });
        const data = await res.json();
        if (data.status === 'success') {
            const img = document.getElementById('videoStream');
            img.src = '/video_feed?t=' + new Date().getTime();
        }
    } catch (err) {
        console.error('Error switching source:', err);
    }
}

// Reload parking positions
async function reloadPositions() {
    try {
        const res = await fetch('/api/reload_positions', { method: 'POST' });
        const data = await res.json();
        alert(`Positions reloaded! Total slots: ${data.total_slots}`);
    } catch (err) {
        alert('Failed to reload positions.');
    }
}

// Trigger background model retrain
async function retrainModel() {
    if (!confirm('Start training the CNN model on your dataset in background?')) return;
    try {
        const res = await fetch('/api/retrain', { method: 'POST' });
        const data = await res.json();
        alert(data.message || 'Retraining initiated.');
    } catch (err) {
        alert('Failed to start training.');
    }
}

// On Page Load
document.addEventListener('DOMContentLoaded', () => {
    initChart();
    updateMetrics();
    setInterval(updateMetrics, 1000); // 1-second polling
});
