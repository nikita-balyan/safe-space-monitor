// Dashboard JavaScript for Safe Space Monitor
class SensorDashboard {
    constructor() {
        this.currentView = 'child';
        this.sensorChart = null;
        this.alerts = [];
        this.isOnline = true;
        this.startTime = Date.now();
        this.readingsCount = 0;
        
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.setupChart();
        this.startDataPolling();
        this.updateSystemInfo();
        
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
    }

    setupChart() {
        const ctx = document.getElementById('sensorChart');
        if (!ctx) return;

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
                        fill: false
                    },
                    {
                        label: 'Light (lux/100)',
                        data: [],
                        borderColor: '#f59e0b',
                        backgroundColor: 'rgba(245, 158, 11, 0.1)',
                        tension: 0.4,
                        fill: false
                    },
                    {
                        label: 'Motion',
                        data: [],
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        tension: 0.4,
                        fill: false
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
                        }
                    },
                    y: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Value'
                        },
                        min: 0,
                        max: 120
                    }
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'top'
                    },
                    tooltip: {
                        enabled: true,
                        mode: 'index'
                    }
                },
                animation: {
                    duration: 750,
                    easing: 'easeInOutQuart'
                }
            }
        });
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
            const response = await fetch('/api/predict');
            if (!response.ok) return null;
            
            return await response.json();
        } catch (error) {
            console.error('Failed to fetch prediction:', error);
            return null;
        }
    }

    updateChildView(data) {
        const { noise, light, motion } = data;
        
        // Determine overall status
        const dangerThresholds = { noise: 100, light: 8000, motion: 80 };
        const warningThresholds = { noise: 70, light: 3000, motion: 50 };
        
        let overallStatus = 'good';
        let statusEmoji = '😊';
        let statusText = 'All Good!';
        let statusDetails = 'Everything looks perfect';
        
        // Check for danger conditions
        if (noise > dangerThresholds.noise || light > dangerThresholds.light || motion > dangerThresholds.motion) {
            overallStatus = 'danger';
            statusEmoji = '😟';
            statusText = 'Needs Attention!';
            statusDetails = 'Something might be too much';
        }
        // Check for warning conditions
        else if (noise > warningThresholds.noise || light > warningThresholds.light || motion > warningThresholds.motion) {
            overallStatus = 'warning';
            statusEmoji = '😐';
            statusText = 'Be Careful';
            statusDetails = 'Might be uncomfortable';
        }
        
        // Update main status card
        const statusCard = document.querySelector('.child-status-card');
        const emojiEl = document.getElementById('statusEmoji');
        const textEl = document.getElementById('statusText');
        const detailsEl = document.getElementById('statusDetails');
        
        if (statusCard && emojiEl && textEl && detailsEl) {
            statusCard.className = `child-status-card text-center p-5 rounded-4 shadow-lg ${overallStatus}`;
            emojiEl.textContent = statusEmoji;
            textEl.textContent = statusText;
            detailsEl.textContent = statusDetails;
        }
        
        // Update individual sensor cards
        this.updateChildSensorCard('childNoiseStatus', noise, warningThresholds.noise, dangerThresholds.noise);
        this.updateChildSensorCard('childLightStatus', light, warningThresholds.light, dangerThresholds.light);
        this.updateChildSensorCard('childMotionStatus', motion, warningThresholds.motion, dangerThresholds.motion);
    }

    updateChildSensorCard(elementId, value, warningThreshold, dangerThreshold) {
        const element = document.getElementById(elementId);
        if (!element) return;
        
        const card = element.closest('.child-sensor-card');
        
        if (value > dangerThreshold) {
            element.textContent = 'Too Much';
            card.className = 'child-sensor-card text-center p-3 rounded-3 danger';
        } else if (value > warningThreshold) {
            element.textContent = 'A Bit Much';
            card.className = 'child-sensor-card text-center p-3 rounded-3 warning';
        } else {
            element.textContent = 'Good';
            card.className = 'child-sensor-card text-center p-3 rounded-3';
        }
    }

    updateCaregiverView(data) {
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
    }

    renderAlerts() {
        const alertList = document.getElementById('alertList');
        if (!alertList) return;
        
        if (this.alerts.length === 0) {
            alertList.innerHTML = `
                <div class="text-muted text-center">
                    <i class="fas fa-check-circle fa-2x mb-2"></i>
                    <p>No alerts detected</p>
                </div>
            `;
            return;
        }
        
        alertList.innerHTML = this.alerts.map(alert => `
            <div class="alert-item ${alert.severity}">
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
        const prediction = await this.fetchPrediction();
        const valueEl = document.getElementById('predictionValue');
        const statusEl = document.getElementById('predictionStatus');
        
        if (prediction && valueEl && statusEl) {
            const probability = Math.round(prediction.probability * 100);
            valueEl.textContent = probability;
            
            if (prediction.prediction === 1) {
                statusEl.textContent = 'Overload Risk Detected';
                statusEl.className = 'prediction-status text-danger';
            } else if (probability > 70) {
                statusEl.textContent = 'Elevated Risk';
                statusEl.className = 'prediction-status text-warning';
            } else {
                statusEl.textContent = 'Normal Conditions';
                statusEl.className = 'prediction-status text-success';
            }
        }
    }

    updateChart(historicalData) {
        if (!this.sensorChart || !historicalData.length) return;
        
        // Prepare data for chart
        const labels = historicalData.map(reading => {
            const date = new Date(reading.timestamp);
            return date.toLocaleTimeString();
        }).slice(-30); // Last 30 readings
        
        const noiseData = historicalData.map(reading => reading.noise).slice(-30);
        const lightData = historicalData.map(reading => reading.light / 100).slice(-30); // Scale down light for visibility
        const motionData = historicalData.map(reading => reading.motion).slice(-30);
        
        // Update chart data
        this.sensorChart.data.labels = labels;
        this.sensorChart.data.datasets[0].data = noiseData;
        this.sensorChart.data.datasets[1].data = lightData;
        this.sensorChart.data.datasets[2].data = motionData;
        
        this.sensorChart.update('none'); // Update without animation for real-time feel
    }

    updateSystemInfo() {
        const uptimeEl = document.getElementById('uptimeValue');
        const readingsEl = document.getElementById('readingsCount');
        
        if (uptimeEl) {
            const uptime = Math.floor((Date.now() - this.startTime) / 1000);
            const minutes = Math.floor(uptime / 60);
            const seconds = uptime % 60;
            uptimeEl.textContent = `${minutes}:${seconds.toString().padStart(2, '0')}`;
        }
        
        if (readingsEl) {
            readingsEl.textContent = this.readingsCount;
        }
    }

    updateAlertsCount() {
        const alertsEl = document.getElementById('alertsCount');
        if (alertsEl) {
            alertsEl.textContent = this.alerts.length;
        }
    }

    async startDataPolling() {
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
            
            // Update status indicator
            this.updateStatusIndicator();
            
            // Continue polling
            setTimeout(poll, 2000); // Poll every 2 seconds
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
            <div class="toast align-items-center text-bg-${type} border-0" role="alert">
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
}

// Initialize dashboard when DOM is loaded
function initializeDashboard() {
    new SensorDashboard();
}

// Export for use in templates
window.initializeDashboard = initializeDashboard;
