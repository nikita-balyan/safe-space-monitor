// Dashboard JavaScript for Safe Space Monitor with Real-time Socket.IO Integration
// Enhanced with Interactive Activities, User Profiles, and Advanced Features
class SensorDashboard {
    constructor() {
        this.currentView = 'caregiver';
        this.sensorChart = null;
        this.predictionChart = null;
        this.gaugeChart = null;
        this.alerts = [];
        this.isOnline = true;
        this.startTime = Date.now();
        this.readingsCount = 0;
        this.recommendationsCount = 0;
        this.lastRecommendationTime = 0;
        this.recommendationCooldown = 30000;
        this.currentOverloadType = null;
        this.isInitialized = false;
        this.socket = io(); // Socket.IO connection
        
        // Enhanced properties
        this.enhancedActivities = [];
        this.userProfile = null;
        this.currentActivity = null;
        this.activityTimer = null;
        
        this.thresholds = {
            noise: { warning: 70, danger: 100, max: 120 },
            light: { warning: 3000, danger: 8000, max: 10000 },
            motion: { warning: 50, danger: 80, max: 100 }
        };
        
        // Real-time data buffers
        this.realTimeData = {
            timestamps: [],
            sensorReadings: [],
            predictions: []
        };
    }

    async init() {
        if (this.isInitialized) return;
        
        console.log('🚀 Initializing dashboard...');
        
        // Wait for DOM to be fully ready
        await this.waitForDOM();
        
        this.setupEventListeners();
        this.setupSocketListeners();
        await this.setupCharts();
        await this.loadInitialData();
        this.updateSystemInfo();
        
        // Initialize enhanced systems
        this.setupActivitySystem();
        this.setupUserProfiles();
        
        // Check URL for view parameter
        const urlParams = new URLSearchParams(window.location.search);
        const viewParam = urlParams.get('view');
        if (viewParam && ['child', 'caregiver'].includes(viewParam)) {
            this.switchView(viewParam);
        }
        
        this.isInitialized = true;
        console.log('✅ Dashboard initialized successfully');
    }

    waitForDOM() {
        return new Promise((resolve) => {
            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', resolve);
            } else {
                // Small delay to ensure all elements are rendered
                setTimeout(resolve, 100);
            }
        });
    }

    setupEventListeners() {
        console.log('🔧 Setting up event listeners...');
        
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
            
            if (e.target.closest('.apply-recommendation-btn')) {
                this.handleApplyRecommendation(e);
            }
            
            // Enhanced activity system events
            if (e.target.closest('.activity-card')) {
                const card = e.target.closest('.activity-card');
                const activityId = parseInt(card.dataset.activityId);
                this.openActivityModal(activityId);
            }
            
            if (e.target.id === 'startActivityBtn') {
                if (this.currentActivity) {
                    this.startActivity(this.currentActivity);
                }
            }
        });

        // Activity modal close events
        const activityModal = document.getElementById('activityModal');
        if (activityModal) {
            activityModal.addEventListener('hidden.bs.modal', () => {
                this.currentActivity = null;
            });
        }

        // Breathing canvas click to exit
        const breathingCanvas = document.getElementById('breathingCanvas');
        if (breathingCanvas) {
            breathingCanvas.addEventListener('click', () => {
                if (this.currentActivity) {
                    this.completeActivity(this.currentActivity);
                }
            });
        }

        // Keyboard events for activity system
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                const canvas = document.getElementById('breathingCanvas');
                if (canvas && canvas.style.display === 'block') {
                    canvas.style.display = 'none';
                    this.showToast('Activity ended', 'You can always try again later', 'info');
                }
            }
        });
    }

    // Enhanced Activity System
    setupActivitySystem() {
        console.log('🎯 Setting up activity system...');
        
        // Load enhanced activities
        this.loadEnhancedActivities();
        
        // Setup activity event listeners
        this.setupActivityListeners();
    }

    async loadEnhancedActivities() {
        try {
            const response = await fetch('/api/activities/enhanced');
            if (response.ok) {
                this.enhancedActivities = await response.json();
                this.renderEnhancedActivities();
                console.log('✅ Enhanced activities loaded:', this.enhancedActivities.length);
            }
        } catch (error) {
            console.error('Failed to load enhanced activities:', error);
        }
    }

    renderEnhancedActivities() {
        const container = document.getElementById('activities-container');
        if (!container || !this.enhancedActivities) return;
        
        container.innerHTML = this.enhancedActivities.map(activity => `
            <div class="border rounded-lg p-4 text-center hover:shadow-md transition-shadow cursor-pointer activity-card" 
                 data-activity-id="${activity.id}">
                <div class="text-3xl mb-3">${activity.emoji}</div>
                <h4 class="font-semibold text-lg mb-2">${activity.name}</h4>
                <p class="text-sm text-gray-600 mb-3">${activity.description}</p>
                <div class="flex justify-between items-center text-xs text-gray-500">
                    <span>${Math.floor(activity.duration / 60)}min</span>
                    <span class="px-2 py-1 rounded-full ${this.getActivityTypeColor(activity.type)}">
                        ${activity.type}
                    </span>
                </div>
            </div>
        `).join('');
        
        console.log('✅ Enhanced activities rendered');
    }

    getActivityTypeColor(type) {
        const colors = {
            'breathing': 'bg-blue-100 text-blue-800',
            'sensory': 'bg-purple-100 text-purple-800',
            'physical': 'bg-amber-100 text-amber-800',
            'visual': 'bg-red-100 text-red-800',
            'mental': 'bg-gray-100 text-gray-800'
        };
        return colors[type] || 'bg-gray-100 text-gray-800';
    }

    openActivityModal(activityId) {
        const activity = this.enhancedActivities.find(a => a.id === activityId);
        if (!activity) return;
        
        this.currentActivity = activity;
        
        const modal = new bootstrap.Modal(document.getElementById('activityModal'));
        const title = document.getElementById('activityModalTitle');
        const container = document.getElementById('activityContainer');
        
        title.textContent = activity.name;
        
        container.innerHTML = `
            <div class="text-center mb-4">
                <div class="text-4xl mb-3">${activity.emoji}</div>
                <p class="text-gray-600 mb-4">${activity.description}</p>
            </div>
            
            <div class="bg-gray-50 rounded-lg p-4 mb-4">
                <h6 class="font-semibold mb-2">Instructions:</h6>
                <div id="activityInstructions">
                    ${activity.instructions.map((step, index) => `
                        <div class="flex items-start mb-2 instruction-step" data-step="${index}">
                            <span class="inline-flex items-center justify-center w-6 h-6 bg-blue-500 text-white text-xs rounded-full mr-3 mt-1">${index + 1}</span>
                            <span class="text-sm">${step.text} <small class="text-gray-500">(${step.duration}s)</small></span>
                        </div>
                    `).join('')}
                </div>
            </div>
            
            <div class="grid grid-cols-2 gap-4 text-sm">
                <div class="text-center p-3 bg-white rounded-lg">
                    <div class="font-semibold text-gray-500">Duration</div>
                    <div class="text-lg font-bold">${Math.floor(activity.duration / 60)}:${(activity.duration % 60).toString().padStart(2, '0')}</div>
                </div>
                <div class="text-center p-3 bg-white rounded-lg">
                    <div class="font-semibold text-gray-500">Type</div>
                    <div class="text-lg font-bold capitalize">${activity.type}</div>
                </div>
            </div>
        `;
        
        // Update start button
        const startBtn = document.getElementById('startActivityBtn');
        startBtn.onclick = () => this.startActivity(activity);
        
        modal.show();
    }

    startActivity(activity) {
        console.log('Starting activity:', activity.name);
        
        // Close modal
        const modal = bootstrap.Modal.getInstance(document.getElementById('activityModal'));
        modal.hide();
        
        // Show fullscreen activity
        this.showFullscreenActivity(activity);
    }

    showFullscreenActivity(activity) {
        const canvas = document.getElementById('breathingCanvas');
        const ctx = canvas.getContext('2d');
        
        // Show canvas
        canvas.style.display = 'block';
        
        // Set canvas size
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
        
        let currentStep = 0;
        let stepStartTime = Date.now();
        let isRunning = true;
        
        // Animation function
        const animate = () => {
            if (!isRunning) return;
            
            const currentTime = Date.now();
            const stepElapsed = (currentTime - stepStartTime) / 1000;
            const currentInstruction = activity.instructions[currentStep];
            const stepProgress = stepElapsed / currentInstruction.duration;
            
            // Clear canvas
            ctx.fillStyle = 'rgba(0, 0, 0, 0.9)';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            // Draw activity visualization based on type
            this.drawActivityVisualization(ctx, activity, currentInstruction, stepProgress);
            
            // Draw instructions
            this.drawActivityInstructions(ctx, currentInstruction, stepProgress, stepElapsed, currentInstruction.duration);
            
            // Check if step is complete
            if (stepElapsed >= currentInstruction.duration) {
                currentStep++;
                stepStartTime = currentTime;
                
                if (currentStep >= activity.instructions.length) {
                    // Activity complete
                    this.completeActivity(activity);
                    return;
                }
            }
            
            requestAnimationFrame(animate);
        };
        
        // Start animation
        animate();
    }

    drawActivityVisualization(ctx, activity, instruction, progress) {
        const centerX = ctx.canvas.width / 2;
        const centerY = ctx.canvas.height / 2 - 50;
        const maxSize = Math.min(ctx.canvas.width, ctx.canvas.height) * 0.3;
        
        ctx.fillStyle = activity.color;
        ctx.strokeStyle = activity.color;
        ctx.lineWidth = 4;
        
        switch (activity.animation) {
            case 'circle_breathe':
            case 'ball_breathe':
                // Breathing circle animation
                let size;
                if (instruction.action === 'inhale') {
                    size = maxSize * 0.3 + (maxSize * 0.7 * progress);
                } else if (instruction.action === 'exhale') {
                    size = maxSize - (maxSize * 0.7 * progress);
                } else {
                    size = maxSize * 0.5;
                }
                
                ctx.beginPath();
                ctx.arc(centerX, centerY, size, 0, Math.PI * 2);
                ctx.fill();
                break;
                
            case 'box_breathe':
                // Box breathing animation
                const boxSize = maxSize * 0.8;
                const pulse = Math.sin(progress * Math.PI * 2) * 10;
                
                ctx.beginPath();
                ctx.rect(centerX - boxSize/2, centerY - boxSize/2, boxSize, boxSize);
                ctx.stroke();
                
                // Animated corner
                ctx.beginPath();
                ctx.arc(centerX + boxSize/2 + pulse, centerY - boxSize/2 - pulse, 10, 0, Math.PI * 2);
                ctx.fill();
                break;
                
            case 'counting':
                // Counting animation
                const numberSize = maxSize * 0.5;
                ctx.font = `bold ${numberSize}px Arial`;
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillStyle = 'white';
                
                if (instruction.action.startsWith('count')) {
                    const count = instruction.action.replace('count', '');
                    ctx.fillText(count, centerX, centerY);
                }
                break;
                
            default:
                // Default circle
                ctx.beginPath();
                ctx.arc(centerX, centerY, maxSize * 0.5, 0, Math.PI * 2);
                ctx.stroke();
        }
    }

    drawActivityInstructions(ctx, instruction, progress, elapsed, total) {
        const centerX = ctx.canvas.width / 2;
        const centerY = ctx.canvas.height / 2 + 100;
        
        ctx.fillStyle = 'white';
        ctx.font = '24px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        
        // Current instruction
        ctx.fillText(instruction.text, centerX, centerY);
        
        // Progress bar
        const barWidth = 400;
        const barHeight = 8;
        const barX = centerX - barWidth / 2;
        const barY = centerY + 40;
        
        ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
        ctx.fillRect(barX, barY, barWidth, barHeight);
        
        ctx.fillStyle = this.currentActivity.color;
        ctx.fillRect(barX, barY, barWidth * progress, barHeight);
        
        // Timer
        ctx.font = '18px Arial';
        ctx.fillStyle = 'white';
        ctx.fillText(`${Math.ceil(total - elapsed)}s`, centerX, barY + 30);
    }

    completeActivity(activity) {
        const canvas = document.getElementById('breathingCanvas');
        const ctx = canvas.getContext('2d');
        
        // Show completion screen
        ctx.fillStyle = 'rgba(0, 0, 0, 0.9)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        ctx.fillStyle = 'white';
        ctx.font = 'bold 48px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('🎉 Great Job!', canvas.width / 2, canvas.height / 2 - 50);
        
        ctx.font = '24px Arial';
        ctx.fillText('You completed the activity', canvas.width / 2, canvas.height / 2 + 20);
        ctx.fillText('Click anywhere to continue', canvas.width / 2, canvas.height / 2 + 70);
        
        // Record activity completion
        this.recordActivityCompletion(activity.id);
        
        // Set timeout to auto-close
        setTimeout(() => {
            canvas.style.display = 'none';
            this.showToast('Activity completed', 'Great job completing the calming activity!', 'success');
        }, 3000);
    }

    async recordActivityCompletion(activityId) {
        try {
            await fetch(`/api/activity/${activityId}/start`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    completed: true,
                    timestamp: new Date().toISOString()
                })
            });
            
            // Also record in user profile
            await this.recordActivityCompletionProfile(activityId);
        } catch (error) {
            console.error('Failed to record activity completion:', error);
        }
    }

    async recordActivityCompletionProfile(activityId, rating = 5, durationActual = null) {
        try {
            const activity = this.enhancedActivities.find(a => a.id === activityId);
            if (!activity) return;
            
            const response = await fetch('/api/profile/activity-complete', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    user_id: 'default',
                    activity_id: activityId,
                    activity_type: activity.type,
                    rating: rating,
                    duration_actual: durationActual,
                    timestamp: new Date().toISOString()
                })
            });
            
            if (response.ok) {
                console.log('Activity completion recorded in profile');
            }
        } catch (error) {
            console.error('Failed to record activity completion in profile:', error);
        }
    }

    setupActivityListeners() {
        // Additional activity-specific listeners can be added here
        console.log('✅ Activity listeners setup complete');
    }

    // Enhanced User Profile System
    setupUserProfiles() {
        console.log('👤 Setting up user profiles...');
        
        // Check for existing profile
        this.loadUserProfile();
        
        // Setup profile modal (could be extended)
        this.setupProfileModal();
    }

    async loadUserProfile() {
        try {
            const response = await fetch('/api/profile/enhanced?user_id=default');
            if (response.ok) {
                this.userProfile = await response.json();
                console.log('User profile loaded:', this.userProfile);
                
                // Apply user settings
                this.applyUserSettings();
            }
        } catch (error) {
            console.error('Failed to load user profile:', error);
            // Use default profile
            this.userProfile = {
                profile: {
                    age: 8,
                    preferences: {
                        sensory_preferences: {
                            noise_sensitivity: 'medium',
                            light_sensitivity: 'high',
                            motion_sensitivity: 'low'
                        },
                        preferred_activities: ['breathing', 'visual'],
                        communication_style: 'visual'
                    },
                    settings: {
                        animation_speed: 'normal',
                        sound_effects: true,
                        color_scheme: 'calm'
                    }
                }
            };
        }
    }

    applyUserSettings() {
        const settings = this.userProfile.profile.settings;
        
        // Apply animation speed
        if (settings.animation_speed === 'slow') {
            document.documentElement.style.setProperty('--animation-duration', '0.8s');
        } else if (settings.animation_speed === 'fast') {
            document.documentElement.style.setProperty('--animation-duration', '0.2s');
        }
        
        // Apply color scheme
        if (settings.color_scheme === 'high-contrast') {
            document.body.classList.add('high-contrast');
        }
        
        // Apply reduced motion
        if (settings.reduced_motion) {
            document.body.classList.add('reduced-motion');
        }
        
        console.log('✅ User settings applied');
    }

    setupProfileModal() {
        // This would set up a profile editing modal
        // For now, we'll use the existing profile system
        console.log('✅ Profile modal setup complete');
    }

    async recordStrategyFeedback(strategyId, overloadType, wasEffective) {
        try {
            const response = await fetch('/api/profile/strategy-feedback', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    user_id: 'default',
                    strategy_id: strategyId,
                    overload_type: overloadType,
                    effective: wasEffective
                })
            });
            
            if (response.ok) {
                console.log('Strategy feedback recorded');
            }
        } catch (error) {
            console.error('Failed to record strategy feedback:', error);
        }
    }

    // Existing methods with enhancements
    setupSocketListeners() {
        console.log('🔌 Setting up Socket.IO listeners...');
        
        // Real-time sensor updates
        this.socket.on('sensor_update', (data) => {
            console.log('📡 Real-time update received:', data);
            this.handleRealTimeUpdate(data);
        });

        // Real-time alerts
        this.socket.on('alert', (alertData) => {
            console.log('🚨 Alert received:', alertData);
            this.handleRealTimeAlert(alertData);
        });

        // Connection status
        this.socket.on('connect', () => {
            this.isOnline = true;
            this.showToast('Connected', 'Real-time data streaming active', 'success');
            this.updateConnectionStatus(true);
        });

        this.socket.on('disconnect', () => {
            this.isOnline = false;
            this.showToast('Disconnected', 'Real-time updates paused', 'warning');
            this.updateConnectionStatus(false);
        });

        this.socket.on('connect_error', (error) => {
            console.error('Socket connection error:', error);
            this.isOnline = false;
            this.updateConnectionStatus(false);
        });
    }

    updateConnectionStatus(connected) {
        const statusElement = document.getElementById('connectionStatus');
        if (statusElement) {
            if (connected) {
                statusElement.innerHTML = '<i class="fas fa-circle text-success me-1"></i><span>Connected</span>';
            } else {
                statusElement.innerHTML = '<i class="fas fa-circle text-danger me-1"></i><span>Disconnected</span>';
            }
        }
    }

    handleRealTimeUpdate(data) {
        this.readingsCount++;
        
        // Update sensor displays
        this.updateSensorDisplays(data.sensor_data);
        this.updateRiskDisplay(data.prediction, data.sensor_data);
        
        // Store data for charts
        this.addRealTimeData(data);
        
        // Update charts
        this.updateRealTimeCharts();
        
        // Check for overload and get recommendations
        this.checkForOverloadAndRecommend(data.sensor_data, data.prediction);
        
        // Update system info
        this.updateSystemInfo();
    }

    handleRealTimeAlert(alertData) {
        this.addAlert(alertData.message, alertData.level, alertData.timestamp);
        
        // Show immediate notification
        this.showToast('Alert Triggered', alertData.message, alertData.level === 'high' ? 'error' : 'warning');
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
    }

    async setupCharts() {
        console.log('📊 Setting up charts...');
        
        // Additional delay to ensure DOM is ready
        await new Promise(resolve => setTimeout(resolve, 200));
        
        const sensorChartSuccess = await this.setupSensorChart();
        const predictionChartSuccess = await this.setupPredictionChart();
        const gaugeChartSuccess = await this.setupGaugeChart();
        
        console.log('✅ Charts initialization complete:', {
            sensorChart: sensorChartSuccess,
            predictionChart: predictionChartSuccess,
            gaugeChart: gaugeChartSuccess
        });
    }

    async setupSensorChart() {
        // Try multiple selectors for the chart canvas
        const selectors = ['#sensorChart', 'canvas[data-chart="sensor"]', 'canvas'];
        let ctx = null;
        
        for (const selector of selectors) {
            ctx = document.querySelector(selector);
            if (ctx && ctx.getContext) {
                console.log(`✅ Found chart canvas with selector: ${selector}`);
                break;
            }
        }
        
        if (!ctx) {
            console.error('❌ Sensor chart canvas not found! Available canvases:');
            document.querySelectorAll('canvas').forEach((canvas, index) => {
                console.log(`  Canvas ${index}:`, canvas.id, canvas.className);
            });
            return false;
        }

        console.log('📈 Initializing sensor chart...');

        // Destroy existing chart if it exists
        if (this.sensorChart) {
            this.sensorChart.destroy();
        }

        try {
            this.sensorChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        {
                            label: 'Noise (dB)',
                            data: [],
                            borderColor: '#EF4444',
                            backgroundColor: 'rgba(239, 68, 68, 0.1)',
                            tension: 0.4,
                            fill: true,
                            borderWidth: 2,
                            pointRadius: 2,
                            pointHoverRadius: 5
                        },
                        {
                            label: 'Light (lux)',
                            data: [],
                            borderColor: '#F59E0B',
                            backgroundColor: 'rgba(245, 158, 11, 0.1)',
                            tension: 0.4,
                            fill: true,
                            borderWidth: 2,
                            pointRadius: 2,
                            pointHoverRadius: 5
                        },
                        {
                            label: 'Motion',
                            data: [],
                            borderColor: '#10B981',
                            backgroundColor: 'rgba(16, 185, 129, 0.1)',
                            tension: 0.4,
                            fill: true,
                            borderWidth: 2,
                            pointRadius: 2,
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
                            beginAtZero: true
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
                        duration: 500,
                        easing: 'easeOutQuart'
                    }
                }
            });
            console.log('✅ Sensor chart initialized successfully');
            return true;
        } catch (error) {
            console.error('❌ Error initializing sensor chart:', error);
            return false;
        }
    }

    async setupPredictionChart() {
        // Try multiple selectors
        const selectors = ['#predictionChart', 'canvas[data-chart="prediction"]'];
        let ctx = null;
        
        for (const selector of selectors) {
            ctx = document.querySelector(selector);
            if (ctx && ctx.getContext) break;
        }
        
        if (!ctx) {
            console.log('ℹ️ Prediction chart not found in template, skipping...');
            return false;
        }

        console.log('📊 Initializing prediction chart...');

        // Destroy existing chart if it exists
        if (this.predictionChart) {
            this.predictionChart.destroy();
        }

        try {
            this.predictionChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Overload Probability',
                        data: [],
                        borderColor: '#8B5CF6',
                        backgroundColor: 'rgba(139, 92, 246, 0.1)',
                        tension: 0.4,
                        fill: true,
                        borderWidth: 2,
                        pointRadius: 2,
                        pointHoverRadius: 5
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: { 
                            min: 0, 
                            max: 1,
                            ticks: { 
                                callback: value => (value * 100).toFixed(0) + '%' 
                            },
                            title: {
                                display: true,
                                text: 'Probability'
                            }
                        },
                        x: {
                            grid: {
                                display: false
                            },
                            ticks: {
                                maxRotation: 0,
                                autoSkip: true,
                                maxTicksLimit: 8
                            },
                            title: {
                                display: true,
                                text: 'Time'
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            display: true,
                            position: 'top'
                        },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    return `Probability: ${(context.parsed.y * 100).toFixed(1)}%`;
                                }
                            }
                        }
                    }
                }
            });
            console.log('✅ Prediction chart initialized successfully');
            return true;
        } catch (error) {
            console.error('❌ Error initializing prediction chart:', error);
            return false;
        }
    }

    async setupGaugeChart() {
        // Try multiple selectors
        const selectors = ['#predictionGauge', 'canvas[data-chart="gauge"]'];
        let ctx = null;
        
        for (const selector of selectors) {
            ctx = document.querySelector(selector);
            if (ctx && ctx.getContext) break;
        }
        
        if (!ctx) {
            console.error('❌ Gauge chart canvas not found!');
            return false;
        }
        
        console.log('🎯 Initializing gauge chart...');

        // Destroy existing chart if it exists
        if (this.gaugeChart) {
            this.gaugeChart.destroy();
        }

        try {
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
            console.log('✅ Gauge chart initialized successfully');
            return true;
        } catch (error) {
            console.error('❌ Error initializing gauge chart:', error);
            return false;
        }
    }

    addRealTimeData(data) {
        const timestamp = new Date().toLocaleTimeString();
        
        // Add to data buffers
        this.realTimeData.timestamps.push(timestamp);
        this.realTimeData.sensorReadings.push(data.sensor_data);
        this.realTimeData.predictions.push(data.prediction);
        
        // Keep only last 20 data points for performance
        if (this.realTimeData.timestamps.length > 20) {
            this.realTimeData.timestamps.shift();
            this.realTimeData.sensorReadings.shift();
            this.realTimeData.predictions.shift();
        }
        
        console.log('📊 Data buffer updated:', {
            timestamps: this.realTimeData.timestamps.length,
            readings: this.realTimeData.sensorReadings.length,
            predictions: this.realTimeData.predictions.length
        });
    }

    updateRealTimeCharts() {
        // Check if charts are ready
        if (!this.sensorChart) {
            console.log('⚠️ Sensor chart not ready for update');
            return;
        }
        
        const timestamps = this.realTimeData.timestamps;
        const sensorReadings = this.realTimeData.sensorReadings;
        const predictions = this.realTimeData.predictions;
        
        if (timestamps.length === 0) {
            console.log('ℹ️ No data available for charts');
            return;
        }
        
        console.log('🔄 Updating charts with:', timestamps.length, 'data points');
        
        try {
            // Update sensor chart
            this.sensorChart.data.labels = timestamps;
            this.sensorChart.data.datasets[0].data = sensorReadings.map(r => r.noise);
            this.sensorChart.data.datasets[1].data = sensorReadings.map(r => r.light);
            this.sensorChart.data.datasets[2].data = sensorReadings.map(r => r.motion);
            this.sensorChart.update('none');
            
            // Update prediction chart if it exists
            if (this.predictionChart) {
                this.predictionChart.data.labels = timestamps;
                this.predictionChart.data.datasets[0].data = predictions;
                this.predictionChart.update('none');
            }
            
            console.log('✅ Charts updated successfully');
        } catch (error) {
            console.error('❌ Error updating charts:', error);
        }
    }

    updateSensorDisplays(sensorData) {
        console.log('🔧 Updating sensor displays:', sensorData);
        
        // Update main sensor displays - with fallbacks for missing elements
        this.updateElementText('noiseValue', Math.round(sensorData.noise));
        this.updateElementText('lightValue', Math.round(sensorData.light));
        this.updateElementText('motionValue', Math.round(sensorData.motion));
        
        // Update child view displays
        this.updateElementText('childNoiseValue', Math.round(sensorData.noise));
        this.updateElementText('childLightValue', Math.round(sensorData.light));
        this.updateElementText('childMotionValue', Math.round(sensorData.motion));
        
        // Update progress bars and status
        this.updateSensorWidget('noise', sensorData.noise, this.thresholds.noise, this.thresholds.noise.max);
        this.updateSensorWidget('light', sensorData.light, this.thresholds.light, this.thresholds.light.max);
        this.updateSensorWidget('motion', sensorData.motion, this.thresholds.motion, this.thresholds.motion.max);
        
        // Update child sensor status
        this.updateChildSensorCard('childNoiseStatus', sensorData.noise, this.thresholds.noise.warning, this.thresholds.noise.danger);
        this.updateChildSensorCard('childLightStatus', sensorData.light, this.thresholds.light.warning, this.thresholds.light.danger);
        this.updateChildSensorCard('childMotionStatus', sensorData.motion, this.thresholds.motion.warning, this.thresholds.motion.danger);
    }

    updateElementText(elementId, value) {
        const element = document.getElementById(elementId);
        if (element) {
            element.textContent = value;
        } else {
            // Don't log missing elements that are optional
            const optionalElements = ['riskValue', 'predictionValue'];
            if (!optionalElements.includes(elementId)) {
                console.log(`Element not found: ${elementId}`);
            }
        }
    }

    updateRiskDisplay(probability, sensorData) {
        console.log('📈 Updating risk display:', probability);
        
        // Update risk value (if element exists)
        this.updateElementText('riskValue', (probability * 100).toFixed(1) + '%');
        this.updateElementText('gaugePercentage', (probability * 100).toFixed(0) + '%');
        
        // Update gauge chart
        if (this.gaugeChart) {
            const probabilityPercent = probability * 100;
            this.gaugeChart.data.datasets[0].data = [probabilityPercent, 100 - probabilityPercent];
            
            // Update gauge color based on risk level
            let gaugeColor;
            if (probability > 0.7) {
                gaugeColor = '#dc3545'; // Red for high risk
            } else if (probability > 0.4) {
                gaugeColor = '#ffc107'; // Yellow for medium risk
            } else {
                gaugeColor = '#28a745'; // Green for low risk
            }
            
            this.gaugeChart.data.datasets[0].backgroundColor = [gaugeColor, '#e9ecef'];
            this.gaugeChart.update('none');
        }
        
        // Update prediction status
        const predictionStatus = document.getElementById('predictionStatus');
        if (predictionStatus) {
            if (probability > 0.7) {
                predictionStatus.textContent = 'High Overload Risk';
                predictionStatus.className = 'prediction-status text-danger fw-bold';
            } else if (probability > 0.4) {
                predictionStatus.textContent = 'Moderate Risk';
                predictionStatus.className = 'prediction-status text-warning fw-bold';
            } else {
                predictionStatus.textContent = 'Normal Conditions';
                predictionStatus.className = 'prediction-status text-success';
            }
        }
        
        // Update overload status badge
        const overloadStatus = document.getElementById('overloadStatus');
        if (overloadStatus) {
            if (probability > 0.7) {
                overloadStatus.textContent = 'High Risk';
                overloadStatus.className = 'badge bg-danger';
            } else if (probability > 0.4) {
                overloadStatus.textContent = 'Moderate Risk';
                overloadStatus.className = 'badge bg-warning';
            } else {
                overloadStatus.textContent = 'Normal';
                overloadStatus.className = 'badge bg-success';
            }
        }
        
        // Update child view status
        this.updateChildStatus(probability, sensorData);
    }

    updateChildStatus(probability, sensorData) {
        const statusCard = document.getElementById('childStatusCard');
        const emojiEl = document.getElementById('statusEmoji');
        const textEl = document.getElementById('statusText');
        const detailsEl = document.getElementById('statusDetails');
        const suggestionEl = document.getElementById('childSuggestion');
        
        if (!statusCard || !emojiEl || !textEl || !detailsEl) {
            console.log('Child view elements not found');
            return;
        }
        
        // Add animation classes
        statusCard.classList.add('fade-in');
        emojiEl.classList.add('emoji-transition');
        
        if (probability > 0.7) {
            // High risk
            statusCard.className = 'child-view-card text-center p-5 rounded-4 shadow-lg status-danger fade-in';
            emojiEl.textContent = '😰';
            textEl.textContent = 'Too Much!';
            detailsEl.textContent = 'Need a break';
            
            if (suggestionEl) {
                suggestionEl.innerHTML = `
                    <i class="fas fa-volume-mute me-2 text-warning"></i>
                    <strong>Let's find a quiet space</strong>
                `;
                suggestionEl.style.display = 'block';
            }
        } else if (probability > 0.4) {
            // Medium risk
            statusCard.className = 'child-view-card text-center p-5 rounded-4 shadow-lg status-warning fade-in';
            emojiEl.textContent = '😟';
            textEl.textContent = 'A Bit Much';
            detailsEl.textContent = 'Getting overwhelmed';
            
            if (suggestionEl) {
                suggestionEl.innerHTML = `
                    <i class="fas fa-wind me-2 text-warning"></i>
                    <strong>Try some deep breaths</strong>
                `;
                suggestionEl.style.display = 'block';
            }
        } else {
            // Low risk
            statusCard.className = 'child-view-card text-center p-5 rounded-4 shadow-lg status-good fade-in';
            emojiEl.textContent = '😊';
            textEl.textContent = 'All Good!';
            detailsEl.textContent = 'Everything looks perfect';
            
            if (suggestionEl) {
                suggestionEl.style.display = 'none';
            }
        }
        
        // Remove animation classes after animation completes
        setTimeout(() => {
            statusCard.classList.remove('fade-in');
            emojiEl.classList.remove('emoji-transition');
        }, 1000);
    }

    updateSensorWidget(type, value, thresholds, maxValue) {
        const progressEl = document.getElementById(`${type}Progress`);
        
        if (progressEl) {
            const percentage = Math.min((value / maxValue) * 100, 100);
            progressEl.style.width = `${percentage}%`;
            
            // Update color based on thresholds
            if (value > thresholds.danger) {
                progressEl.className = 'progress-bar bg-danger';
            } else if (value > thresholds.warning) {
                progressEl.className = 'progress-bar bg-warning';
            } else {
                progressEl.className = 'progress-bar bg-success';
            }
        }
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

    async checkForOverloadAndRecommend(sensorData, prediction) {
        const overloadType = this.checkForOverload(sensorData);
        const now = Date.now();
        
        if (overloadType && now - this.lastRecommendationTime > this.recommendationCooldown) {
            this.currentOverloadType = overloadType;
            this.lastRecommendationTime = now;
            
            try {
                const response = await fetch('/api/current');
                if (response.ok) {
                    const data = await response.json();
                    if (data.recommendations) {
                        this.displayRecommendations(data.recommendations, overloadType);
                    }
                }
            } catch (error) {
                console.error('Failed to fetch recommendations:', error);
            }
        } else if (!overloadType && this.currentOverloadType) {
            this.currentOverloadType = null;
            this.clearRecommendations();
        }
    }

    checkForOverload(data) {
        const { noise, light, motion } = data;
        
        if (noise > this.thresholds.noise.danger) return 'auditory';
        if (light > this.thresholds.light.danger) return 'visual';
        if (motion > this.thresholds.motion.danger) return 'motion';
        return null;
    }

    async displayRecommendations(recommendations, overloadType) {
        const container = document.getElementById('recommendationsList');
        const countElement = document.getElementById('recommendationsCount');
        const recommendationsCountValue = document.getElementById('recommendationsCountValue');
        
        if (!recommendations || recommendations.length === 0) {
            container.innerHTML = `
                <div class="text-center py-4">
                    <i class="fas fa-info-circle fa-2x text-muted mb-2"></i>
                    <p class="text-muted">No recommendations available</p>
                </div>
            `;
            if (countElement) countElement.textContent = '0';
            if (recommendationsCountValue) recommendationsCountValue.textContent = '0';
            return;
        }
        
        this.recommendationsCount = recommendations.length;
        if (countElement) countElement.textContent = this.recommendationsCount;
        if (recommendationsCountValue) recommendationsCountValue.textContent = this.recommendationsCount;
        
        let html = `
            <div class="mb-3">
                <small class="text-muted">AI recommendations for ${overloadType} overload:</small>
            </div>
        `;
        
        recommendations.forEach((rec, index) => {
            const effectiveness = rec.effectiveness || 75;
            const priority = rec.priority || 'medium';
            
            html += `
                <div class="card mb-3 strategy-card ${overloadType} fade-in" data-recommendation-index="${index}">
                    <div class="card-body p-3">
                        <div class="d-flex align-items-start">
                            <div class="flex-grow-1">
                                <div class="d-flex justify-content-between align-items-start mb-2">
                                    <h6 class="card-title mb-0">${rec.title}</h6>
                                    <span class="badge ${priority === 'high' ? 'bg-danger' : 'bg-warning'}">${priority}</span>
                                </div>
                                <p class="card-text small text-muted mb-2">${rec.description}</p>
                                
                                <div class="d-flex justify-content-between align-items-center">
                                    <div>
                                        <span class="badge bg-light text-dark small">
                                            <i class="fas fa-chart-line me-1 text-info"></i>
                                            ${effectiveness}% effective
                                        </span>
                                    </div>
                                    <div class="btn-group" role="group">
                                        <button type="button" class="btn btn-sm btn-primary apply-recommendation-btn">
                                            <i class="fas fa-play me-1"></i> Try
                                        </button>
                                        <button type="button" class="btn btn-sm btn-outline-success feedback-btn" data-helpful="true">
                                            <i class="fas fa-thumbs-up"></i>
                                        </button>
                                        <button type="button" class="btn btn-sm btn-outline-danger feedback-btn" data-helpful="false">
                                            <i class="fas fa-thumbs-down"></i>
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

    clearRecommendations() {
        const container = document.getElementById('recommendationsList');
        const countElement = document.getElementById('recommendationsCount');
        const recommendationsCountValue = document.getElementById('recommendationsCountValue');
        
        container.innerHTML = `
            <div class="text-center py-4">
                <i class="fas fa-check-circle fa-2x text-success mb-2"></i>
                <p class="text-success">Environment is comfortable</p>
                <small class="text-muted">No recommendations needed at this time</small>
            </div>
        `;
        
        this.recommendationsCount = 0;
        if (countElement) countElement.textContent = '0';
        if (recommendationsCountValue) recommendationsCountValue.textContent = '0';
    }

    addAlert(message, severity, timestamp) {
        const alert = { 
            message, 
            severity, 
            timestamp: timestamp || new Date().toISOString(), 
            id: Date.now() 
        };
        
        // Avoid duplicate alerts
        const isDuplicate = this.alerts.some(existing => 
            existing.message === message && 
            Date.now() - existing.id < 5000
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
                <div class="text-muted text-center py-4">
                    <i class="fas fa-check-circle fa-2x mb-2"></i>
                    <p>No alerts detected</p>
                </div>
            `;
            return;
        }
        
        alertList.innerHTML = this.alerts.map(alert => `
            <div class="alert-item ${alert.severity} mb-2 slide-in">
                <div class="d-flex justify-content-between align-items-start">
                    <div>
                        <strong>${alert.severity === 'high' ? '🚨 High Risk' : '⚠️ Warning'}</strong>
                        <div class="small">${alert.message}</div>
                    </div>
                    <small class="text-muted">${new Date(alert.timestamp).toLocaleTimeString()}</small>
                </div>
            </div>
        `).join('');
    }

    async handleStrategyFeedback(event) {
        const button = event.target.closest('.feedback-btn');
        if (!button) return;
        
        const strategyCard = button.closest('.strategy-card');
        const strategyIndex = strategyCard.dataset.recommendationIndex;
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
        await this.recordFeedback(strategyIndex, wasHelpful);
        
        // Record in user profile
        if (this.currentOverloadType) {
            await this.recordStrategyFeedback(`strategy_${strategyIndex}`, this.currentOverloadType, wasHelpful);
        }
    }

    async handleApplyRecommendation(event) {
        const button = event.target.closest('.apply-recommendation-btn');
        if (!button) return;
        
        const strategyCard = button.closest('.strategy-card');
        const strategyIndex = strategyCard.dataset.recommendationIndex;
        
        // Visual feedback
        button.classList.remove('btn-primary');
        button.classList.add('btn-success');
        button.innerHTML = '<i class="fas fa-check me-1"></i> Applied';
        button.disabled = true;
        
        // Send feedback
        await fetch('/api/feedback', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                recommendation: `Strategy ${strategyIndex}`,
                action: 'applied'
            })
        });
        
        this.showToast('Strategy Applied', 'Thank you for trying this recommendation', 'success');
    }

    async recordFeedback(strategyIndex, wasHelpful) {
        try {
            const response = await fetch('/api/feedback', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    strategy_id: `strategy_${strategyIndex}`,
                    helpful: wasHelpful
                })
            });
            
            if (response.ok) {
                this.showToast('Feedback Recorded', 'Thank you for your feedback!', 'success');
            }
        } catch (error) {
            console.error('Failed to record feedback:', error);
        }
    }

    updateSystemInfo() {
        const uptimeEl = document.getElementById('uptimeValue');
        const readingsEl = document.getElementById('readingsCountValue');
        const alertsEl = document.getElementById('alertsCountValue');
        const recommendationsEl = document.getElementById('recommendationsCountValue');
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
        
        if (recommendationsEl) {
            recommendationsEl.textContent = this.recommendationsCount;
        }
        
        if (systemStatusEl && statusIndicatorEl) {
            systemStatusEl.textContent = this.isOnline ? 'Real-time Data Streaming' : 'Connection Lost';
            statusIndicatorEl.className = `fas fa-circle ${this.isOnline ? 'text-success' : 'text-danger'} me-2`;
        }
    }

    updateAlertsCount() {
        const alertsEl = document.getElementById('alertsCount');
        const alertsCountValue = document.getElementById('alertsCountValue');
        
        if (alertsEl) {
            alertsEl.textContent = this.alerts.length;
            alertsEl.className = `badge ${this.alerts.length > 0 ? 'bg-danger' : 'bg-secondary'}`;
        }
        
        if (alertsCountValue) {
            alertsCountValue.textContent = this.alerts.length;
        }
    }

    async loadInitialData() {
        try {
            console.log('📥 Loading initial data...');
            
            // Load current data
            const response = await fetch('/api/current');
            if (response.ok) {
                const data = await response.json();
                this.handleRealTimeUpdate(data);
            }
            
            // Load historical data for charts
            const historyResponse = await fetch('/api/history');
            if (historyResponse.ok) {
                const historyData = await historyResponse.json();
                this.initializeChartsWithHistory(historyData);
            }
            
            this.showToast('System Ready', 'Real-time monitoring activated', 'success');
        } catch (error) {
            console.error('Initial data load failed:', error);
            this.showToast('Connection Error', 'Unable to load initial data', 'error');
        }
    }

    initializeChartsWithHistory(historyData) {
        if (!historyData.sensor_readings || !historyData.predictions) {
            console.log('No historical data available');
            return;
        }
        
        console.log('Initializing charts with historical data:', historyData);
        
        // Initialize real-time data buffers with historical data
        this.realTimeData.timestamps = historyData.sensor_readings.map(r => 
            new Date(r.timestamp).toLocaleTimeString()
        ).slice(-20);
        
        this.realTimeData.sensorReadings = historyData.sensor_readings.slice(-20);
        this.realTimeData.predictions = historyData.predictions.slice(-20).map(p => p.probability);
        
        this.updateRealTimeCharts();
    }

    showToast(title, message, type = 'info') {
        if (typeof window.showToast === 'function') {
            window.showToast(message, type, title);
        } else {
            console.log(`[${type.toUpperCase()}] ${title}: ${message}`);
        }
    }

    async exportData() {
        try {
            // For now, export current real-time data
            const dataStr = JSON.stringify(this.realTimeData, null, 2);
            const blob = new Blob([dataStr], { type: 'application/json' });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `sensor_data_${new Date().toISOString().slice(0, 10)}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
            
            this.showToast('Export Complete', 'Sensor data downloaded as JSON', 'success');
        } catch (error) {
            console.error('Export failed:', error);
            this.showToast('Export Failed', 'Unable to download data', 'error');
        }
    }

    manualRefresh() {
        this.showToast('Refreshing', 'Updating sensor data', 'info');
        this.loadInitialData();
    }

    clearAlerts() {
        this.alerts = [];
        this.renderAlerts();
        this.updateAlertsCount();
        this.showToast('Alerts Cleared', 'All alerts removed', 'info');
    }

    destroy() {
        if (this.sensorChart) {
            this.sensorChart.destroy();
            this.sensorChart = null;
        }
        if (this.predictionChart) {
            this.predictionChart.destroy();
            this.predictionChart = null;
        }
        if (this.gaugeChart) {
            this.gaugeChart.destroy();
            this.gaugeChart = null;
        }
        if (this.socket) {
            this.socket.disconnect();
        }
        
        // Clear activity timer
        if (this.activityTimer) {
            clearTimeout(this.activityTimer);
        }
        
        this.isInitialized = false;
    }
}

// Initialize dashboard
async function initializeDashboard() {
    console.log('🚀 Starting dashboard initialization...');
    
    // Clean up any existing dashboard instance
    if (window.sensorDashboard) {
        window.sensorDashboard.destroy();
    }
    
    // Create new dashboard instance
    window.sensorDashboard = new SensorDashboard();
    await window.sensorDashboard.init();
    
    console.log('🎉 Dashboard initialization complete!');
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