// Dashboard JavaScript for Safe Space Monitor with Recommendation System
class SensorDashboard {
    constructor() {
        this.currentView = 'caregiver';
        this.sensorChart = null;
        this.gaugeChart = null;
        this.alerts = [];
        this.isOnline = true;
        this.startTime = Date.now();
        this.readingsCount = 0;
        this.pollingInterval = null;
        this.lastRecommendationTime = 0;
        this.recommendationCooldown = 30000; // 30 seconds between recommendations
        this.currentOverloadType = null;
        this.historicalData = [];
        this.isInitialized = false;
        
        this.thresholds = {
            noise: { warning: 70, danger: 100, max: 120 },
            light: { warning: 3000, danger: 8000, max: 10000 },
            motion: { warning: 50, danger: 80, max: 100 }
        };
        
        this.init();
    }

    init() {
        if (this.isInitialized) return;
        
        this.setupEventListeners();
        this.setupChart();
        this.setupGaugeChart();
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
        
        this.isInitialized = true;
    }

    setupEventListeners() {
        // View switching
        const childViewBtn = document.getElementById('childViewBtn');
        const caregiverViewBtn = document.getElementById('caregiverViewBtn');
        
        if (childViewBtn && caregiverViewBtn) {
            childViewBtn.addEventListener('click', () => {
                this.switchView('child');
            });
            
            caregiverViewBtn.addEventListener('click', () => {
                this.switchView('caregiver');
            });
        }

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

        // Add event delegation for strategy feedback buttons
        document.addEventListener('click', (e) => {
            if (e.target.closest('.feedback-btn')) {
                this.handleStrategyFeedback(e);
            }
        });
    }

    switchView(view) {
        if (this.currentView === view) return;
        
        this.currentView = view;
        
        // Update button states
        const childViewBtn = document.getElementById('childViewBtn');
        const caregiverViewBtn = document.getElementById('caregiverViewBtn');
        
        if (childViewBtn && caregiverViewBtn) {
            childViewBtn.classList.toggle('active', view === 'child');
            caregiverViewBtn.classList.toggle('active', view === 'caregiver');
        }
        
        // Update view visibility
        const childView = document.getElementById('childView');
        const caregiverView = document.getElementById('caregiverView');
        
        if (childView && caregiverView) {
            childView.classList.toggle('d-none', view !== 'child');
            caregiverView.classList.toggle('d-none', view !== 'caregiver');
        }
        
        // Update URL without reload
        const url = new URL(window.location);
        url.searchParams.set('view', view);
        window.history.replaceState({}, '', url);
        
        // Re-setup chart when switching to caregiver view
        if (view === 'caregiver') {
            setTimeout(() => {
                this.setupChart();
                this.updateChart(this.historicalData);
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

        // Check if the canvas is already being used by another chart instance
        if (Chart.instances && Chart.instances.length > 0) {
            Chart.instances.forEach(instance => {
                if (instance.canvas && instance.canvas.id === 'sensorChart') {
                    instance.destroy();
                }
            });
        }

        this.sensorChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Noise (dB)',
                        data: [],
                        borderColor: '#0d6efd',
                        backgroundColor: 'rgba(13, 110, 253, 0.1)',
                        tension: 0.4,
                        fill: true,
                        borderWidth: 2,
                        pointRadius: 0,
                        pointHoverRadius: 5
                    },
                    {
                        label: 'Light (lux/100)',
                        data: [],
                        borderColor: '#ffc107',
                        backgroundColor: 'rgba(255, 193, 7, 0.1)',
                        tension: 0.4,
                        fill: true,
                        borderWidth: 2,
                        pointRadius: 0,
                        pointHoverRadius: 5
                    },
                    {
                        label: 'Motion',
                        data: [],
                        borderColor: '#198754',
                        backgroundColor: 'rgba(25, 135, 84, 0.1)',
                        tension: 0.4,
                        fill: true,
                        borderWidth: 2,
                        pointRadius: 0,
                        pointHoverRadius: 5
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
                        },
                        ticks: {
                            maxRotation: 0,
                            autoSkip: true,
                            maxTicksLimit: 8
                        }
                    },
                    y: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Value'
                        },
                        beginAtZero: true,
                        suggestedMax: 120
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

    setupGaugeChart() {
        const ctx = document.getElementById('predictionGauge');
        if (!ctx) return;
        
        // Destroy existing chart if it exists
        if (this.gaugeChart) {
            this.gaugeChart.destroy();
            this.gaugeChart = null;
        }

        this.gaugeChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                datasets: [{
                    data: [0, 100],
                    backgroundColor: [
                        '#28a745', // Green
                        '#e9ecef'  // Gray background
                    ],
                    borderWidth: 0,
                    circumference: 180,
                    rotation: 270
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                cutout: '70%',
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        enabled: false
                    }
                },
                animation: {
                    animateRotate: true,
                    animateScale: true
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
            this.showToast('Connection Error', 'Unable to fetch sensor data', 'error');
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

    async fetchPrediction(sensorData) {
        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    noise: sensorData?.noise || 0,
                    light: sensorData?.light || 0,
                    motion: sensorData?.motion || 0
                })
            });
        
            if (!response.ok) {
                console.error('Prediction API error:', response.status);
                return null;
            }
        
            return await response.json();
        } catch (error) {
            console.error('Failed to fetch prediction:', error);
            return null;
        }
    }

    async getRecommendations(overloadType) {
        try {
            const now = Date.now();
            if (now - this.lastRecommendationTime < this.recommendationCooldown) {
                return []; // Skip if in cooldown period
            }

            const response = await fetch(`/api/recommendations?type=${overloadType}`);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            
            const data = await response.json();
            this.lastRecommendationTime = now;
            
            return data.recommendations || [];
        } catch (error) {
            console.error('Failed to fetch recommendations:', error);
            return [];
        }
    }

    async recordFeedback(strategyId, wasHelpful) {
        try {
            const response = await fetch('/api/feedback', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    strategy_id: strategyId,
                    helpful: wasHelpful
                })
            });
            
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            
            const result = await response.json();
            this.showToast('Feedback Recorded', 'Thank you for your feedback!', 'success');
            return result;
        } catch (error) {
            console.error('Failed to record feedback:', error);
            this.showToast('Feedback Error', 'Could not record feedback', 'error');
            return null;
        }
    }

    async updateChildView(data) {
        if (!data) return;
        
        const { noise, light, motion } = data;
        
        // Update sensor values display
        this.updateSensorValueDisplay('childNoiseValue', noise);
        this.updateSensorValueDisplay('childLightValue', light);
        this.updateSensorValueDisplay('childMotionValue', motion);
        
        // Update sensor status
        this.updateChildSensorCard('childNoiseStatus', noise, this.thresholds.noise.warning, this.thresholds.noise.danger);
        this.updateChildSensorCard('childLightStatus', light, this.thresholds.light.warning, this.thresholds.light.danger);
        this.updateChildSensorCard('childMotionStatus', motion, this.thresholds.motion.warning, this.thresholds.motion.danger);
        
        // Check for overload and update main status
        const overloadType = this.checkForOverload(data);
        if (overloadType) {
            this.updateChildStatus('warning', '😟', 'Getting overwhelmed', `${overloadType} level is high`);
            
            // Get and show recommendation for child
            const recommendations = await this.getRecommendations(overloadType);
            if (recommendations.length > 0) {
                this.showChildSuggestion(recommendations[0]);
            }
        } else {
            this.updateChildStatus('good', '😊', 'All Good!', 'Everything looks perfect');
            this.hideChildSuggestion();
        }
    }

    updateChildStatus(status, emoji, text, details) {
        const statusCard = document.getElementById('childStatusCard');
        const emojiEl = document.getElementById('statusEmoji');
        const textEl = document.getElementById('statusText');
        const detailsEl = document.getElementById('statusDetails');
        
        if (statusCard && emojiEl && textEl && detailsEl) {
            // Add animation classes
            statusCard.classList.add('fade-in');
            emojiEl.classList.add('emoji-transition');
            
            statusCard.className = `child-view-card text-center p-5 rounded-4 shadow-lg status-${status} fade-in`;
            emojiEl.textContent = emoji;
            textEl.textContent = text;
            detailsEl.textContent = details;
            
            // Remove animation classes after animation completes
            setTimeout(() => {
                statusCard.classList.remove('fade-in');
                emojiEl.classList.remove('emoji-transition');
            }, 1000);
        }
    }

    showChildSuggestion(strategy) {
        const suggestionEl = document.getElementById('childSuggestion');
        if (suggestionEl) {
            suggestionEl.innerHTML = `
                <i class="fas ${this.getStrategyIcon(strategy)} me-2 text-warning"></i>
                <strong>Try this:</strong> ${strategy.name}
            `;
            suggestionEl.style.display = 'block';
        }
    }

    hideChildSuggestion() {
        const suggestionEl = document.getElementById('childSuggestion');
        if (suggestionEl) {
            suggestionEl.style.display = 'none';
        }
    }

    getStrategyIcon(strategy) {
        const iconMap = {
            'noise_cancelling_headphones': 'fa-headphones',
            'white_noise': 'fa-wave-square',
            'calming_music': 'fa-music',
            'dim_lights': 'fa-lightbulb',
            'blue_light_filter': 'fa-glasses',
            'deep_pressure': 'fa-weight-hanging',
            'deep_breathing': 'fa-wind',
            'counting_exercise': 'fa-calculator',
            'default': 'fa-lightbulb'
        };
        return iconMap[strategy.id] || iconMap.default;
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

    checkForOverload(data) {
        const { noise, light, motion } = data;
        
        if (noise > this.thresholds.noise.danger) return 'auditory';
        if (light > this.thresholds.light.danger) return 'visual';
        if (motion > this.thresholds.motion.danger) return 'motion';
        return null;
    }

    async updateCaregiverView(data) {
        if (!data) return;
        
        const { noise, light, motion } = data;
        
        // Update sensor values
        this.updateSensorWidget('noise', noise, this.thresholds.noise, this.thresholds.noise.max);
        this.updateSensorWidget('light', light, this.thresholds.light, this.thresholds.light.max);
        this.updateSensorWidget('motion', motion, this.thresholds.motion, this.thresholds.motion.max);
        
        // Check for alerts
        this.checkAndAddAlerts(data);
        
        // Update prediction
        await this.updatePrediction();
        
        // Check for overload and get recommendations
        const overloadType = this.checkForOverload(data);
        if (overloadType && overloadType !== this.currentOverloadType) {
            this.currentOverloadType = overloadType;
            const recommendations = await this.getRecommendations(overloadType);
            this.displayRecommendations(recommendations, overloadType);
            this.showToast('Overload Detected', `${overloadType} levels are high. Showing recommendations.`, 'warning');
        } else if (!overloadType && this.currentOverloadType) {
            this.currentOverloadType = null;
            this.clearRecommendations();
        }
    }

    updateSensorWidget(type, value, thresholds, maxValue) {
        const valueEl = document.getElementById(`${type}Value`);
        const progressEl = document.getElementById(`${type}Progress`);
        const labelEl = document.getElementById(`${type}Label`);
        
        if (valueEl) {
            valueEl.textContent = Math.round(value);
        }
        
        if (progressEl) {
            const percentage = Math.min((value / maxValue) * 100, 100);
            progressEl.style.width = `${percentage}%`;
            
            // Update color based on thresholds
            if (value > thresholds.danger) {
                progressEl.className = 'progress-bar bg-danger';
                if (labelEl) labelEl.className = 'sensor-label text-danger';
            } else if (value > thresholds.warning) {
                progressEl.className = 'progress-bar bg-warning';
                if (labelEl) labelEl.className = 'sensor-label text-warning';
            } else {
                progressEl.className = 'progress-bar bg-success';
                if (labelEl) labelEl.className = 'sensor-label text-success';
            }
        }
    }

    checkAndAddAlerts(data) {
        const { noise, light, motion, timestamp } = data;

        Object.entries({ noise, light, motion }).forEach(([sensor, value]) => {
            const sensorThresholds = this.thresholds[sensor];
            
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
            Date.now() - existing.id < 5000
        );
        
        if (isDuplicate) return;
        
        this.alerts.unshift(alert);
        this.alerts = this.alerts.slice(0, 10);
        
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
            <div class="alert-item alert alert-${alert.severity} mb-2 slide-in">
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
        const gaugePercentageEl = document.getElementById('gaugePercentage');
        
        if (predictionResponse && valueEl && statusEl && widgetEl && gaugePercentageEl) {
            const predictionData = predictionResponse.prediction;
            let probability = predictionData.probability || 0;
            
            if (isNaN(probability)) {
                probability = 0;
            }
            
            const probabilityPercent = Math.round(probability * 100);
            valueEl.textContent = probabilityPercent;
            gaugePercentageEl.textContent = `${probabilityPercent}%`;
            
            // Update gauge chart
            if (this.gaugeChart) {
                this.gaugeChart.data.datasets[0].data = [probabilityPercent, 100 - probabilityPercent];
                
                // Update gauge color based on risk level
                let gaugeColor;
                if (predictionData.prediction === 1) {
                    gaugeColor = '#dc3545'; // Red for high risk
                } else if (probabilityPercent > 70) {
                    gaugeColor = '#ffc107'; // Yellow for medium risk
                } else {
                    gaugeColor = '#28a745'; // Green for low risk
                }
                
                this.gaugeChart.data.datasets[0].backgroundColor = [gaugeColor, '#e9ecef'];
                this.gaugeChart.update('none');
            }
            
            // Update prediction status
            if (predictionData.prediction === 1) {
                statusEl.textContent = 'Overload Risk';
                statusEl.className = 'prediction-status text-danger fw-bold';
                widgetEl.className = 'prediction-widget high-risk';
            } else if (probabilityPercent > 70) {
                statusEl.textContent = 'High Risk';
                statusEl.className = 'prediction-status text-warning fw-bold';
                widgetEl.className = 'prediction-widget medium-risk';
            } else if (probabilityPercent > 30) {
                statusEl.textContent = 'Moderate Risk';
                statusEl.className = 'prediction-status text-warning';
                widgetEl.className = 'prediction-widget medium-risk';
            } else {
                statusEl.textContent = 'Normal Conditions';
                statusEl.className = 'prediction-status text-success';
                widgetEl.className = 'prediction-widget low-risk';
            }
        }
    }

    async displayRecommendations(recommendations, overloadType) {
        const container = document.getElementById('recommendationsList');
        const countElement = document.getElementById('recommendationsCount');
        
        if (!recommendations || recommendations.length === 0) {
            container.innerHTML = `
                <div class="text-center py-4">
                    <i class="fas fa-info-circle fa-2x text-muted mb-2"></i>
                    <p class="text-muted">No recommendations available</p>
                </div>
            `;
            countElement.textContent = '0';
            return;
        }
        
        countElement.textContent = recommendations.length;
        
        let html = `
            <div class="mb-3">
                <small class="text-muted">For ${overloadType} overload:</small>
            </div>
        `;
        
        recommendations.forEach((strategy) => {
            const successRate = Math.round((strategy.feedback_score || 0.5) * 100);
            const emoji = strategy.emoji || '💡';
            const feedbackCount = strategy.feedback_count || 0;
            
            // Determine rating stars based on success rate
            const stars = this.generateRatingStars(successRate);
            
            html += `
                <div class="card mb-3 strategy-card ${overloadType} fade-in" data-strategy-id="${strategy.id}">
                    <div class="card-body p-3">
                        <div class="d-flex align-items-start">
                            <span class="fs-4 me-3">${emoji}</span>
                            <div class="flex-grow-1">
                                <h6 class="card-title mb-1">${strategy.name}</h6>
                                <p class="card-text small text-muted mb-2">${strategy.description}</p>
                                
                                <!-- Feedback history -->
                                <div class="feedback-history mb-2">
                                    <small class="text-muted">
                                        <i class="fas fa-chart-line me-1"></i>
                                        ${successRate}% success rate (${feedbackCount} responses)
                                    </small>
                                    <div class="rating-stars small">
                                        ${stars}
                                    </div>
                                </div>
                                
                                <div class="d-flex justify-content-between align-items-center">
                                    <div>
                                        <span class="badge bg-light text-dark small">
                                            <i class="fas fa-star me-1 text-warning"></i>
                                            ${successRate}% helpful
                                        </span>
                                    </div>
                                    <div class="btn-group strategy-feedback" role="group">
                                        <button type="button" class="btn btn-sm btn-outline-success feedback-btn" data-helpful="true">
                                            <i class="fas fa-thumbs-up"></i> Helpful
                                        </button>
                                        <button type="button" class="btn btn-sm btn-outline-danger feedback-btn" data-helpful="false">
                                            <i class="fas fa-thumbs-down"></i> Not Helpful
                                        </button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        });
        
        container.innerHTML = html;
    }

    generateRatingStars(successRate) {
        const fullStars = Math.floor(successRate / 20);
        const halfStar = successRate % 20 >= 10 ? 1 : 0;
        const emptyStars = 5 - fullStars - halfStar;
        
        let stars = '';
        
        // Add full stars
        for (let i = 0; i < fullStars; i++) {
            stars += '<i class="fas fa-star text-warning"></i>';
        }
        
        // Add half star
        if (halfStar) {
            stars += '<i class="fas fa-star-half-alt text-warning"></i>';
        }
        
        // Add empty stars
        for (let i = 0; i < emptyStars; i++) {
            stars += '<i class="far fa-star text-warning"></i>';
        }
        
        return stars;
    }

    clearRecommendations() {
        const container = document.getElementById('recommendationsList');
        const countElement = document.getElementById('recommendationsCount');
        
        container.innerHTML = `
            <div class="text-center py-4">
                <i class="fas fa-check-circle fa-2x text-success mb-2"></i>
                <p class="text-success">No overload detected</p>
                <small class="text-muted">Sensor levels are normal</small>
            </div>
        `;
        countElement.textContent = '0';
    }

    async handleStrategyFeedback(event) {
        const button = event.target.closest('.feedback-btn');
        if (!button) return;
        
        const strategyCard = button.closest('.strategy-card');
        const strategyId = strategyCard.dataset.strategyId;
        const wasHelpful = button.dataset.helpful === 'true';
        
        // Visual feedback
        if (wasHelpful) {
            button.classList.remove('btn-outline-success');
            button.classList.add('btn-success');
        } else {
            button.classList.remove('btn-outline-danger');
            button.classList.add('btn-danger');
        }
        
        // Disable both buttons
        strategyCard.querySelectorAll('.feedback-btn').forEach(btn => {
            btn.disabled = true;
        });
        
        // Record feedback
        await this.recordFeedback(strategyId, wasHelpful);
    }

    updateChart(historicalData) {
        if (!this.sensorChart || !historicalData || !historicalData.length) return;
        
        // Store historical data for later use
        this.historicalData = historicalData;
        
        // Show only last 10 minutes of data (adjust as needed)
        const now = Date.now();
        const tenMinutesAgo = now - (10 * 60 * 1000);
    
        const recentData = historicalData.filter(reading => {
            return new Date(reading.timestamp).getTime() > tenMinutesAgo;
        }).slice(-30); // Show max 30 points
    
        const labels = recentData.map(reading => {
            return new Date(reading.timestamp).toLocaleTimeString();
        });
        
        const noiseData = recentData.map(reading => reading.noise);
        const lightData = recentData.map(reading => reading.light / 100);
        const motionData = recentData.map(reading => reading.motion);
    
        this.sensorChart.data.labels = labels;
        this.sensorChart.data.datasets[0].data = noiseData;
        this.sensorChart.data.datasets[1].data = lightData;
        this.sensorChart.data.datasets[2].data = motionData;
    
        this.sensorChart.update('none');
    
        // Add visual indicators for thresholds
        this.addChartThresholds();
    }

    addChartThresholds() {
        if (!this.sensorChart) return;
    
        // Add threshold lines to the chart
        const chart = this.sensorChart;
        const yScale = chart.scales.y;
    
        // Remove existing threshold lines if any
        chart.options.plugins.annotation = chart.options.plugins.annotation || {};
        chart.options.plugins.annotation.annotations = [];
    
        // Add noise threshold lines
        chart.options.plugins.annotation.annotations.push(
            {
                type: 'line',
                mode: 'horizontal',
                scaleID: 'y',
                value: this.thresholds.noise.warning,
                borderColor: 'orange',
                borderWidth: 1,
                borderDash: [5, 5],
                label: {
                    enabled: true,
                    content: 'Noise Warning'
                }
            },
            {
                type: 'line',
                mode: 'horizontal',
                scaleID: 'y',
                value: this.thresholds.noise.danger,
                borderColor: 'red',
                borderWidth: 2,
                label: {
                    enabled: true,
                    content: 'Noise Danger'
                }
            }
        );
    
        chart.update('none');
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
            systemStatusEl.textContent = this.isOnline ? 'System Online' : 'Connection Lost';
            statusIndicatorEl.className = `fas fa-circle ${this.isOnline ? 'text-success' : 'text-danger'} me-2`;
        }
    }

    updateAlertsCount() {
        const alertsEl = document.getElementById('alertsCount');
        if (alertsEl) {
            alertsEl.textContent = this.alerts.length;
            alertsEl.className = `badge ${this.alerts.length > 0 ? 'bg-danger' : 'bg-secondary'}`;
        }
    }

    async startDataPolling() {
        if (this.pollingInterval) {
            clearInterval(this.pollingInterval);
        }
        
        const poll = async () => {
            const currentData = await this.fetchSensorData();
            
            if (currentData) {
                if (this.currentView === 'child') {
                    this.updateChildView(currentData);
                } else {
                    this.updateCaregiverView(currentData);
                }
                
                // Only fetch historical data for caregiver view
                if (this.currentView === 'caregiver') {
                    const historicalData = await this.fetchHistoricalData();
                    this.updateChart(historicalData);
                }
            }
            
            this.updateSystemInfo();
            this.updateAlertsCount();
        };
        
        // Initial poll
        await poll();
        
        // Set up interval polling
        this.pollingInterval = setInterval(poll, 2000);
    }

    showToast(title, message, type = 'info') {
        // Use the global showToast function from base.html
        if (typeof window.showToast === 'function') {
            window.showToast(title, message, type);
        } else {
            // Fallback to console log
            console.log(`${title}: ${message}`);
        }
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
            
            this.showToast('Export Complete', 'Sensor data downloaded', 'success');
        } catch (error) {
            console.error('Export failed:', error);
            this.showToast('Export Failed', 'Unable to download data', 'error');
        }
    }

    manualRefresh() {
        this.showToast('Refreshing', 'Updating sensor data', 'info');
        this.startDataPolling();
    }

    clearAlerts() {
        this.alerts = [];
        this.renderAlerts();
        this.updateAlertsCount();
        this.showToast('Alerts Cleared', 'All alerts removed', 'info');
    }

    destroy() {
        if (this.pollingInterval) {
            clearInterval(this.pollingInterval);
            this.pollingInterval = null;
        }
        if (this.sensorChart) {
            this.sensorChart.destroy();
            this.sensorChart = null;
        }
        if (this.gaugeChart) {
            this.gaugeChart.destroy();
            this.gaugeChart = null;
        }
        this.isInitialized = false;
    }
}

// Initialize dashboard
function initializeDashboard() {
    // Clean up any existing chart instances before creating a new one
    if (window.sensorDashboard) {
        window.sensorDashboard.destroy();
    }
    
    window.sensorDashboard = new SensorDashboard();
}

// Export for global access
window.initializeDashboard = initializeDashboard;
window.SensorDashboard = SensorDashboard;

// Auto-initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeDashboard);
} else {
    initializeDashboard();
}