// Dashboard JavaScript for Safe Space Monitor
class SensorDashboard {
    constructor() {
        this.currentView = 'child';
        this.sensorChart = null;
        this.alerts = [];
        this.isOnline = true;
        this.startTime = Date.now();
        this.readingsCount = 0;
        this.pollingInterval = null;
        
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.setupChart();
        this.startDataPolling();
        this.updateSystemInfo();
        
        // Check URL for view parameter
        const urlParams = new URLSearchParams(window.location.search);
        const viewParam = urlParams.get('view');
        if (viewParam && ['child', 'caregiver'].includes(viewParam)) {
            this.switchView(viewParam);
        }
        
        // Show initial toast
        this.showToast('System Active', 'Sensors are online and monitoring', 'success');
    }

    setupEventListeners() {
        // View switching
        document.getElementById('childViewBtn').addEventListener('click', () => {
            this.switchView('child');
        });
        
        document.getElementById('caregiverViewBtn').addEventListener('click', () => {
            this.switchView('caregiver');
        });

        // Export data
        const exportBtn = document.getElementById('exportDataBtn');
        if (exportBtn) {
            exportBtn.addEventListener('click', () => {
                this.exportData();
            });
        }

        // Manual refresh button
        const refreshBtn = document.getElementById('refreshDataBtn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => {
                this.manualRefresh();
            });
        }

        // Clear alerts button
        const clearAlertsBtn = document.getElementById('clearAlertsBtn');
        if (clearAlertsBtn) {
            clearAlertsBtn.addEventListener('click', () => {
                this.clearAlerts();
            });
        }
    }

    switchView(view) {
        this.currentView = view;
        
        // Update button states
        document.getElementById('childViewBtn').classList.toggle('active', view === 'child');
        document.getElementById('caregiverViewBtn').classList.toggle('active', view === 'caregiver');
        
        // Update view visibility
        document.getElementById('childView').classList.toggle('d-none', view !== 'child');
        document.getElementById('caregiverView').classList.toggle('d-none', view !== 'caregiver');
        
        // Update URL without reload
        const url = new URL(window.location);
        url.searchParams.set('view', view);
        window.history.replaceState({}, '', url);
        
        // Re-setup chart when switching to caregiver view
        if (view === 'caregiver') {
            setTimeout(() => {
                this.setupChart();
                this.startDataPolling();
            }, 100);
        }
    }

    setupChart() {
        const ctx = document.getElementById('sensorChart');
        if (!ctx) return;

        // Destroy existing chart if it exists
        if (this.sensorChart) {
            this.sensorChart.destroy();
            this.sensorChart = null;
        }

        // Check if canvas context is still valid
        if (ctx && ctx.getContext) {
            this.sensorChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        {
                            label: 'Noise (dB)',
                            data: [],
                            borderColor: '#3b82f6',
                            backgroundColor: 'rgba(59, 130, 246, 0.1)',
                            tension: 0.4,
                            fill: false,
                            borderWidth: 2
                        },
                        {
                            label: 'Light (lux/100)',
                            data: [],
                            borderColor: '#f59e0b',
                            backgroundColor: 'rgba(245, 158, 11, 0.1)',
                            tension: 0.4,
                            fill: false,
                            borderWidth: 2
                        },
                        {
                            label: 'Motion',
                            data: [],
                            borderColor: '#10b981',
                            backgroundColor: 'rgba(16, 185, 129, 0.1)',
                            tension: 0.4,
                            fill: false,
                            borderWidth: 2
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {
                        intersect: false,
                        mode: 'index'
                    },
                    scales: {
                        x: {
                            display: true,
                            title: {
                                display: true,
                                text: 'Time'
                            },
                            grid: {
                                display: false
                            }
                        },
                        y: {
                            display: true,
                            title: {
                                display: true,
                                text: 'Value'
                            },
                            min: 0,
                            max: 120,
                            ticks: {
                                stepSize: 20
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            display: true,
                            position: 'top',
                            labels: {
                                usePointStyle: true,
                                padding: 20
                            }
                        },
                        tooltip: {
                            enabled: true,
                            mode: 'index',
                            callbacks: {
                                label: function(context) {
                                    let label = context.dataset.label || '';
                                    if (label) {
                                        label += ': ';
                                    }
                                    if (context.parsed.y !== null) {
                                        label += context.parsed.y.toFixed(1);
                                    }
                                    return label;
                                }
                            }
                        }
                    },
                    animation: {
                        duration: 750,
                        easing: 'easeInOutQuart'
                    }
                }
            });
        }
    }

    async fetchSensorData() {
        try {
            const response = await fetch('/api/current');
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            
            const data = await response.json();
            this.isOnline = true;
            this.readingsCount++;
            
            return data;
        } catch (error) {
            console.error('Failed to fetch sensor data:', error);
            this.isOnline = false;
            this.showToast('Connection Error', 'Unable to fetch sensor data', 'danger');
            return null;
        }
    }

    async fetchHistoricalData() {
        try {
            const response = await fetch('/api/history');
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            
            return await response.json();
        } catch (error) {
            console.error('Failed to fetch historical data:', error);
            return [];
        }
    }

    async fetchPrediction() {
        try {
            // First get current data
            const currentData = await this.fetchSensorData();
            if (!currentData) return null;
            
            // Use the main prediction endpoint
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    noise: currentData.noise,
                    light: currentData.light,
                    motion: currentData.motion
                })
            });
            
            if (!response.ok) {
                console.error('Prediction API error:', response.status);
                return null;
            }
            
            const predictionData = await response.json();
            console.log('Prediction API response:', predictionData);
            return predictionData;
        } catch (error) {
            console.error('Failed to fetch prediction:', error);
            return null;
        }
    }

    async updateChildView(data) {
        if (!data) return;
        
        const { noise, light, motion } = data;
        
        // Get AI prediction for enhanced child view
        const predictionResponse = await this.fetchPrediction();
        
        // Determine overall status using both threshold-based and AI prediction
        const dangerThresholds = { noise: 100, light: 8000, motion: 80 };
        const warningThresholds = { noise: 70, light: 3000, motion: 50 };
        
        let overallStatus = 'good';
        let statusEmoji = '😊';
        let statusText = 'All Good!';
        let statusDetails = 'Everything looks perfect';
        
        // Check for danger conditions (sensors or AI prediction)
        let aiDanger = false;
        let aiHighRisk = false;
        
        if (predictionResponse && predictionResponse.prediction) {
            const predictionData = predictionResponse.prediction;
            aiDanger = predictionData.prediction === 1;
            aiHighRisk = predictionData.probability > 0.7;
        }
        
        if (noise > dangerThresholds.noise || light > dangerThresholds.light || motion > dangerThresholds.motion || aiDanger) {
            overallStatus = 'danger';
            statusEmoji = '😟';
            statusText = 'Needs Attention!';
            statusDetails = aiDanger ? 'Smart monitor says be careful' : 'Something might be too much';
        }
        // Check for warning conditions (sensors or AI prediction)
        else if (noise > warningThresholds.noise || light > warningThresholds.light || motion > warningThresholds.motion || aiHighRisk) {
            overallStatus = 'warning';
            statusEmoji = '😐';
            statusText = 'Be Careful';
            statusDetails = aiHighRisk ? 'Smart monitor says watch out' : 'Might be uncomfortable';
        }
        
        // Update main status card
        const statusCard = document.getElementById('childStatusCard');
        const emojiEl = document.getElementById('statusEmoji');
        const textEl = document.getElementById('statusText');
        const detailsEl = document.getElementById('statusDetails');
        
        if (statusCard && emojiEl && textEl && detailsEl) {
            statusCard.className = `child-status-card text-center p-5 rounded-4 shadow-lg status-${overallStatus}`;
            emojiEl.textContent = statusEmoji;
            textEl.textContent = statusText;
            detailsEl.textContent = statusDetails;
        }
        
        // Update individual sensor cards
        this.updateChildSensorCard('childNoiseStatus', noise, warningThresholds.noise, dangerThresholds.noise);
        this.updateChildSensorCard('childLightStatus', light, warningThresholds.light, dangerThresholds.light);
        this.updateChildSensorCard('childMotionStatus', motion, warningThresholds.motion, dangerThresholds.motion);
        
        // Update sensor values display
        this.updateSensorValueDisplay('childNoiseValue', noise);
        this.updateSensorValueDisplay('childLightValue', light);
        this.updateSensorValueDisplay('childMotionValue', motion);
    }

    updateChildSensorCard(elementId, value, warningThreshold, dangerThreshold) {
        const element = document.getElementById(elementId);
        if (!element) return;
        
        const card = element.closest('.child-sensor-card');
        if (!card) return;
        
        if (value > dangerThreshold) {
            element.textContent = 'Too Much';
            card.className = 'child-sensor-card text-center p-3 rounded-3 status-danger';
        } else if (value > warningThreshold) {
            element.textContent = 'A Bit Much';
            card.className = 'child-sensor-card text-center p-3 rounded-3 status-warning';
        } else {
            element.textContent = 'Good';
            card.className = 'child-sensor-card text-center p-3 rounded-3 status-good';
        }
    }

    updateSensorValueDisplay(elementId, value) {
        const element = document.getElementById(elementId);
        if (element) {
            element.textContent = Math.round(value);
        }
    }

    updateCaregiverView(data) {
        if (!data) return;
        
        const { noise, light, motion } = data;
        
        // Update sensor values
        this.updateSensorWidget('noise', noise, { warning: 70, danger: 100 }, 120);
        this.updateSensorWidget('light', light, { warning: 3000, danger: 8000 }, 10000);
        this.updateSensorWidget('motion', motion, { warning: 50, danger: 80 }, 100);
        
        // Check for alerts
        this.checkAndAddAlerts(data);
        
        // Update prediction
        this.updatePrediction();
    }

    updateSensorWidget(type, value, thresholds, maxValue) {
        const valueEl = document.getElementById(`${type}Value`);
        const progressEl = document.getElementById(`${type}Progress`);
        
        if (valueEl) {
            valueEl.textContent = Math.round(value);
        }
        
        if (progressEl) {
            const percentage = Math.min((value / maxValue) * 100, 100);
            progressEl.style.width = `${percentage}%`;
            
            // Update color based on thresholds
            progressEl.className = 'progress-bar';
            if (value > thresholds.danger) {
                progressEl.classList.add('bg-danger');
            } else if (value > thresholds.warning) {
                progressEl.classList.add('bg-warning');
            } else {
                progressEl.classList.add('bg-success');
            }
        }
    }

    checkAndAddAlerts(data) {
        const { noise, light, motion, timestamp } = data;
        const thresholds = {
            noise: { warning: 70, danger: 100 },
            light: { warning: 3000, danger: 8000 },
            motion: { warning: 50, danger: 80 }
        };

        // Check each sensor for alerts
        Object.entries({ noise, light, motion }).forEach(([sensor, value]) => {
            const sensorThresholds = thresholds[sensor];
            
            if (value > sensorThresholds.danger) {
                this.addAlert(`${sensor.toUpperCase()} CRITICAL: ${Math.round(value)}`, 'danger', timestamp);
            } else if (value > sensorThresholds.warning) {
                this.addAlert(`${sensor.toUpperCase()} Warning: ${Math.round(value)}`, 'warning', timestamp);
            }
        });
    }

    addAlert(message, severity, timestamp) {
        const alert = { message, severity, timestamp, id: Date.now() };
        
        // Avoid duplicate alerts
        const isDuplicate = this.alerts.some(existing => 
            existing.message === message && 
            Date.now() - existing.id < 5000 // 5 second debounce
        );
        
        if (isDuplicate) return;
        
        this.alerts.unshift(alert);
        this.alerts = this.alerts.slice(0, 10); // Keep only last 10 alerts
        
        this.renderAlerts();
        this.updateAlertsCount();
        
        // Show toast for new alerts
        if (this.alerts.length > 0 && this.alerts[0].id === alert.id) {
            this.showToast('New Alert', message, severity);
        }
    }

    renderAlerts() {
        const alertList = document.getElementById('alertList');
        if (!alertList) return;
        
        if (this.alerts.length === 0) {
            alertList.innerHTML = `
                <div class="text-muted text-center py-4">
                    <i class="fas fa-check-circle fa-2x mb-2"></i>
                    <p>No alerts detected</p>
                </div>
            `;
            return;
        }
        
        alertList.innerHTML = this.alerts.map(alert => `
            <div class="alert-item alert-${alert.severity}">
                <div class="d-flex justify-content-between align-items-start">
                    <div>
                        <i class="fas fa-${alert.severity === 'danger' ? 'exclamation-triangle' : 'exclamation-circle'} me-2"></i>
                        ${alert.message}
                    </div>
                    <small class="text-muted">
                        ${new Date(alert.timestamp).toLocaleTimeString()}
                    </small>
                </div>
            </div>
        `).join('');
    }

    async updatePrediction() {
        const predictionResponse = await this.fetchPrediction();
        const valueEl = document.getElementById('predictionValue');
        const statusEl = document.getElementById('predictionStatus');
        const widgetEl = document.getElementById('predictionWidget');
        
        if (predictionResponse && valueEl && statusEl && widgetEl) {
            // Extract the prediction data from the response
            const predictionData = predictionResponse.prediction;
            
            // Handle NaN or invalid probability values
            let probability = predictionData.probability;
            if (isNaN(probability) || !isFinite(probability)) {
                console.warn('Invalid probability value:', probability);
                probability = 0; // Default to 0 if invalid
            }
            
            const probabilityPercent = Math.round(probability * 100);
            valueEl.textContent = probabilityPercent;
            
            // Remove existing risk classes
            widgetEl.classList.remove('high-risk', 'medium-risk', 'low-risk');
            
            // Handle prediction
            const predictionValue = predictionData.prediction;
            
            if (predictionValue === 1) {
                statusEl.textContent = 'Overload Risk Detected';
                statusEl.className = 'prediction-status text-danger';
                widgetEl.classList.add('high-risk');
            } else if (probabilityPercent > 70) {
                statusEl.textContent = 'Elevated Risk';
                statusEl.className = 'prediction-status text-warning';
                widgetEl.classList.add('medium-risk');
            } else if (probabilityPercent > 30) {
                statusEl.textContent = 'Low Risk';
                statusEl.className = 'prediction-status text-warning';
                widgetEl.classList.add('medium-risk');
            } else {
                statusEl.textContent = 'Normal Conditions';
                statusEl.className = 'prediction-status text-success';
                widgetEl.classList.add('low-risk');
            }
        } else {
            // Handle case where prediction is not available
            if (valueEl) valueEl.textContent = 'N/A';
            if (statusEl) {
                statusEl.textContent = 'Prediction Unavailable';
                statusEl.className = 'prediction-status text-muted';
            }
            if (widgetEl) {
                widgetEl.classList.remove('high-risk', 'medium-risk', 'low-risk');
            }
        }
    }

    updateChart(historicalData) {
        if (!this.sensorChart || !historicalData || !historicalData.length) return;
        
        // Prepare data for chart
        const labels = historicalData.map(reading => {
            const date = new Date(reading.timestamp);
            return date.toLocaleTimeString();
        }).slice(-30); // Last 30 readings
        
        const noiseData = historicalData.map(reading => reading.noise).slice(-30);
        const lightData = historicalData.map(reading => reading.light / 100).slice(-30);
        const motionData = historicalData.map(reading => reading.motion).slice(-30);
        
        // Update chart data
        this.sensorChart.data.labels = labels;
        this.sensorChart.data.datasets[0].data = noiseData;
        this.sensorChart.data.datasets[1].data = lightData;
        this.sensorChart.data.datasets[2].data = motionData;
        
        this.sensorChart.update('none');
    }

    updateSystemInfo() {
        const uptimeEl = document.getElementById('uptimeValue');
        const readingsEl = document.getElementById('readingsCountValue');
        const alertsEl = document.getElementById('alertsCountValue');
        const systemStatusEl = document.getElementById('systemStatusText');
        const statusIndicatorEl = document.getElementById('statusIndicator');
        
        if (uptimeEl) {
            const uptime = Math.floor((Date.now() - this.startTime) / 1000);
            const hours = Math.floor(uptime / 3600);
            const minutes = Math.floor((uptime % 3600) / 60);
            const seconds = uptime % 60;
            uptimeEl.textContent = `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        }
        
        if (readingsEl) {
            readingsEl.textContent = this.readingsCount.toLocaleString();
        }
        
        if (alertsEl) {
            alertsEl.textContent = this.alerts.length;
        }
        
        if (systemStatusEl && statusIndicatorEl) {
            if (this.isOnline) {
                systemStatusEl.textContent = 'System Online';
                statusIndicatorEl.className = 'fas fa-circle text-success me-2';
            } else {
                systemStatusEl.textContent = 'Connection Lost';
                statusIndicatorEl.className = 'fas fa-circle text-danger me-2';
            }
        }
    }

    updateAlertsCount() {
        const alertsEl = document.getElementById('alertsCount');
        if (alertsEl) {
            alertsEl.textContent = this.alerts.length;
            alertsEl.className = this.alerts.length > 0 ? 'badge bg-danger' : 'badge bg-secondary';
        }
    }

    async startDataPolling() {
        // Clear any existing polling
        if (this.pollingInterval) {
            clearTimeout(this.pollingInterval);
        }
        
        const poll = async () => {
            const currentData = await this.fetchSensorData();
            
            if (currentData) {
                this.updateChildView(currentData);
                this.updateCaregiverView(currentData);
                
                // Update chart with historical data
                const historicalData = await this.fetchHistoricalData();
                this.updateChart(historicalData);
            }
            
            this.updateSystemInfo();
            this.updateAlertsCount();
            
            // Update status indicator
            this.updateStatusIndicator();
            
            // Continue polling
            this.pollingInterval = setTimeout(poll, 2000); // Poll every 2 seconds
        };
        
        poll();
    }

    updateStatusIndicator() {
        const statusIcon = document.querySelector('#statusToast .fas');
        const statusMessage = document.getElementById('statusMessage');
        
        if (statusIcon && statusMessage) {
            if (this.isOnline) {
                statusIcon.className = 'fas fa-circle text-success me-2';
                statusMessage.textContent = 'Sensors Active';
            } else {
                statusIcon.className = 'fas fa-circle text-danger me-2';
                statusMessage.textContent = 'Connection Lost';
            }
        }
    }

    showToast(title, message, type = 'info') {
        // Create and show Bootstrap toast
        const toastHtml = `
            <div class="toast align-items-center text-bg-${type} border-0" role="alert" data-bs-delay="5000">
                <div class="d-flex">
                    <div class="toast-body">
                        <strong>${title}</strong><br>${message}
                    </div>
                    <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
                </div>
            </div>
        `;
        
        // Add to toast container or create one
        let container = document.getElementById('toast-container');
        if (!container) {
            container = document.createElement('div');
            container.id = 'toast-container';
            container.className = 'toast-container position-fixed bottom-0 end-0 p-3';
            document.body.appendChild(container);
        }
        
        container.insertAdjacentHTML('beforeend', toastHtml);
        
        const toastElement = container.lastElementChild;
        const toast = new bootstrap.Toast(toastElement);
        toast.show();
        
        // Remove from DOM after hiding
        toastElement.addEventListener('hidden.bs.toast', () => {
            toastElement.remove();
        });
    }

    async exportData() {
        try {
            const response = await fetch('/api/export/csv');
            if (!response.ok) throw new Error('Export failed');
            
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `sensor_data_${new Date().toISOString().slice(0, 10)}.csv`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
            
            this.showToast('Export Complete', 'Sensor data has been downloaded', 'success');
        } catch (error) {
            console.error('Export failed:', error);
            this.showToast('Export Failed', 'Unable to export data', 'danger');
        }
    }

    manualRefresh() {
        this.showToast('Refreshing', 'Manually refreshing sensor data', 'info');
        this.startDataPolling();
    }

    clearAlerts() {
        this.alerts = [];
        this.renderAlerts();
        this.updateAlertsCount();
        this.showToast('Alerts Cleared', 'All alerts have been cleared', 'info');
    }

    // Clean up when needed
    destroy() {
        if (this.pollingInterval) {
            clearTimeout(this.pollingInterval);
        }
        if (this.sensorChart) {
            this.sensorChart.destroy();
        }
    }
}

// Initialize dashboard when DOM is loaded
function initializeDashboard() {
    // Wait for Bootstrap to be loaded
    if (typeof bootstrap === 'undefined') {
        console.error('Bootstrap is not loaded');
        return;
    }
    
    window.sensorDashboard = new SensorDashboard();
}

// Export for use in templates
window.initializeDashboard = initializeDashboard;

// Initialize when DOM is fully loaded
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeDashboard);
} else {
    initializeDashboard();
}