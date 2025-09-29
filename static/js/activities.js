// activities.js - Enhanced Calming Activities with TTS and Animations
// Replaces breathing.js with comprehensive activity management

class ActivityManager {
    constructor() {
        this.currentActivity = null;
        this.activitySession = null;
        this.isActivityActive = false;
        this.isPaused = false;
        this.currentStep = 0;
        this.timeRemaining = 0;
        this.totalDuration = 0;
        this.elapsedTime = 0;
        
        // TTS System
        this.speechSynthesis = window.speechSynthesis;
        this.currentUtterance = null;
        this.availableVoices = [];
        
        // Activity settings
        this.settings = {
            voiceOption: 'female_calm',
            speechRate: 1.0,
            volume: 0.8,
            visualTheme: 'default',
            animationSpeed: 'normal',
            hapticFeedback: false
        };
        
        // DOM Elements
        this.elements = {};
        this.timers = {
            activity: null,
            countdown: null,
            animation: null
        };
        
        this.initialize();
    }

    initialize() {
        this.cacheElements();
        this.loadSettings();
        this.loadVoices();
        this.setupEventListeners();
        this.loadActivities();
        
        console.log('🎯 Activity Manager initialized');
    }

    cacheElements() {
        this.elements = {
            // Activity Grid
            activitiesGrid: document.getElementById('activitiesGrid'),
            
            // Voice Settings Modal
            voiceSettingsModal: document.getElementById('voiceSettingsModal'),
            voiceOptions: document.getElementById('voiceOptions'),
            speechRate: document.getElementById('speechRate'),
            speechRateValue: document.getElementById('speechRateValue'),
            volume: document.getElementById('volume'),
            volumeValue: document.getElementById('volumeValue'),
            saveVoiceSettings: document.getElementById('saveVoiceSettings'),
            
            // Activity Session Modal
            activitySessionModal: document.getElementById('activitySessionModal'),
            activitySessionTitle: document.getElementById('activitySessionTitle'),
            activityName: document.getElementById('activityName'),
            activityEmoji: document.getElementById('activityEmoji'),
            animationContainer: document.getElementById('animationContainer'),
            animationIcon: document.getElementById('animationIcon'),
            currentInstruction: document.getElementById('currentInstruction'),
            sessionProgress: document.getElementById('sessionProgress'),
            timeRemaining: document.getElementById('timeRemaining'),
            instructionsList: document.getElementById('instructionsList'),
            
            // Control Buttons
            startActivityBtn: document.getElementById('startActivityBtn'),
            pauseActivityBtn: document.getElementById('pauseActivityBtn'),
            resumeActivityBtn: document.getElementById('resumeActivityBtn'),
            stopActivityBtn: document.getElementById('stopActivityBtn'),
            
            // Header Buttons
            voiceSettingsBtn: document.getElementById('voiceSettingsBtn'),
            activityStatsBtn: document.getElementById('activityStatsBtn')
        };
    }

    setupEventListeners() {
        // Voice Settings
        this.elements.voiceSettingsBtn?.addEventListener('click', () => this.showVoiceSettings());
        this.elements.saveVoiceSettings?.addEventListener('click', () => this.saveVoiceSettings());
        this.elements.speechRate?.addEventListener('input', () => this.updateSpeechRateDisplay());
        this.elements.volume?.addEventListener('input', () => this.updateVolumeDisplay());

        // Activity Controls
        this.elements.startActivityBtn?.addEventListener('click', () => this.startActivitySession());
        this.elements.pauseActivityBtn?.addEventListener('click', () => this.pauseActivitySession());
        this.elements.resumeActivityBtn?.addEventListener('click', () => this.resumeActivitySession());
        this.elements.stopActivityBtn?.addEventListener('click', () => this.stopActivitySession());

        // Delegated event listeners for dynamic content
        document.addEventListener('click', (e) => {
            const startBtn = e.target.closest('.start-activity');
            const detailsBtn = e.target.closest('.view-details');
            const voiceOption = e.target.closest('.voice-option');

            if (startBtn) {
                this.startActivity(startBtn.dataset.activityId);
            } else if (detailsBtn) {
                this.viewActivityDetails(detailsBtn.dataset.activityId);
            } else if (voiceOption) {
                this.selectVoiceOption(voiceOption.dataset.voiceId);
            }
        });

        // Handle modal hidden events
        this.elements.activitySessionModal?.addEventListener('hidden.bs.modal', () => {
            this.cleanupActivity();
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.isActivityActive) {
                this.stopActivitySession();
            }
            if (e.key === ' ' && this.isActivityActive) { // Spacebar
                e.preventDefault();
                if (this.isPaused) {
                    this.resumeActivitySession();
                } else {
                    this.pauseActivitySession();
                }
            }
        });

        // Page visibility changes (pause when tab not active)
        document.addEventListener('visibilitychange', () => {
            if (document.hidden && this.isActivityActive && !this.isPaused) {
                this.pauseActivitySession();
            }
        });
    }

    async loadActivities() {
        try {
            const response = await fetch('/api/activities');
            const activities = await response.json();
            this.displayActivities(activities);
        } catch (error) {
            console.error('Error loading activities:', error);
            this.showToast('Failed to load activities', 'error');
        }
    }

    displayActivities(activities) {
        if (!this.elements.activitiesGrid) return;

        this.elements.activitiesGrid.innerHTML = activities.map(activity => `
            <div class="col-xl-3 col-lg-4 col-md-6">
                <div class="card activity-card h-100">
                    <div class="card-body text-center">
                        <div class="activity-icon" style="color: ${activity.color}">
                            ${activity.emoji}
                        </div>
                        <h5 class="card-title">${activity.name}</h5>
                        <p class="card-text text-muted small">${activity.description}</p>
                        
                        <div class="mb-2">
                            <span class="badge ${this.getDifficultyClass(activity.difficulty)} difficulty-badge">
                                ${activity.difficulty}
                            </span>
                            <span class="badge bg-light text-dark difficulty-badge">
                                ${this.formatDuration(activity.duration)}
                            </span>
                            <span class="badge bg-info difficulty-badge">${activity.age_range}</span>
                        </div>

                        <div class="mb-3">
                            ${activity.benefits.map(benefit => 
                                `<span class="badge bg-light text-dark me-1 mb-1">${benefit}</span>`
                            ).join('')}
                        </div>

                        <div class="accessibility-icons mb-2">
                            ${activity.accessibility.map(access => 
                                `<i class="fas fa-${this.getAccessibilityIcon(access)} me-1 text-muted" title="${access}"></i>`
                            ).join('')}
                        </div>

                        <div class="mt-auto">
                            <button class="btn btn-primary btn-sm start-activity" 
                                    data-activity-id="${activity.id}">
                                <i class="fas fa-play me-1"></i>Start
                            </button>
                            <button class="btn btn-outline-secondary btn-sm ms-1 view-details"
                                    data-activity-id="${activity.id}">
                                <i class="fas fa-info-circle me-1"></i>Details
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `).join('');
    }

    getDifficultyClass(difficulty) {
        const classes = {
            'beginner': 'bg-success',
            'intermediate': 'bg-warning',
            'advanced': 'bg-danger'
        };
        return classes[difficulty] || 'bg-secondary';
    }

    getAccessibilityIcon(accessType) {
        const icons = {
            'visual': 'eye',
            'audio': 'volume-up',
            'haptic': 'hand-paper',
            'tactile': 'hand-holding'
        };
        return icons[accessType] || 'universal-access';
    }

    formatDuration(seconds) {
        if (seconds < 60) return `${seconds}s`;
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = seconds % 60;
        return `${minutes}m${remainingSeconds > 0 ? ` ${remainingSeconds}s` : ''}`;
    }

    async startActivity(activityId) {
        try {
            const response = await fetch('/api/activities');
            const activities = await response.json();
            this.currentActivity = activities.find(a => a.id === activityId);
            
            if (this.currentActivity) {
                this.showActivitySessionModal();
            }
        } catch (error) {
            console.error('Error starting activity:', error);
            this.showToast('Failed to start activity', 'error');
        }
    }

    showActivitySessionModal() {
        if (!this.currentActivity) return;

        // Update modal title
        this.elements.activitySessionTitle.innerHTML = `
            ${this.currentActivity.emoji} ${this.currentActivity.name}
        `;
        this.elements.activityName.textContent = this.currentActivity.name;
        this.elements.activityEmoji.textContent = this.currentActivity.emoji;
        
        // Setup instructions
        this.setupInstructions();
        
        // Reset UI state
        this.resetActivityUI();
        
        // Show modal
        const modal = new bootstrap.Modal(this.elements.activitySessionModal);
        modal.show();
    }

    setupInstructions() {
        if (!this.currentActivity || !this.elements.instructionsList) return;

        this.elements.instructionsList.innerHTML = this.currentActivity.instructions.map((instruction, index) => `
            <div class="instruction-step ${index === 0 ? 'active' : ''}">
                <div class="d-flex align-items-center">
                    <div class="phase-indicator phase-${instruction.phase}"></div>
                    <div>
                        <strong>${instruction.text}</strong>
                        <div class="text-muted small">${instruction.duration} seconds</div>
                    </div>
                </div>
            </div>
        `).join('');

        // Calculate total duration
        this.totalDuration = this.currentActivity.instructions.reduce((sum, inst) => sum + inst.duration, 0);
    }

    async startActivitySession() {
        if (!this.currentActivity) return;

        try {
            const response = await fetch('/api/activities/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    activity_id: this.currentActivity.id,
                    voice_option: this.settings.voiceOption,
                    speech_rate: this.settings.speechRate,
                    volume: this.settings.volume,
                    visual_theme: this.settings.visualTheme
                })
            });

            this.activitySession = await response.json();
            this.isActivityActive = true;
            this.currentStep = 0;
            this.elapsedTime = 0;

            this.showToast('Activity session started', 'success');
            this.updateControlButtons('active');
            this.runActivitySequence();

        } catch (error) {
            console.error('Error starting activity session:', error);
            this.showToast('Failed to start activity session', 'error');
        }
    }

    runActivitySequence() {
        if (!this.currentActivity || !this.isActivityActive) return;

        const executeStep = () => {
            if (this.currentStep >= this.currentActivity.instructions.length || !this.isActivityActive) {
                if (this.isActivityActive) {
                    this.completeActivitySession();
                }
                return;
            }

            const instruction = this.currentActivity.instructions[this.currentStep];
            this.executeInstruction(instruction, this.currentStep);
            
            // Schedule next step
            this.timers.activity = setTimeout(() => {
                if (this.isActivityActive && !this.isPaused) {
                    this.currentStep++;
                    executeStep();
                }
            }, instruction.duration * 1000);
        };

        executeStep();
    }

    executeInstruction(instruction, stepIndex) {
        // Update display
        this.updateActivityDisplay(instruction, stepIndex);
        
        // Speak instruction
        this.speakInstruction(instruction.text);
        
        // Update progress
        this.elapsedTime += instruction.duration;
        const progress = (this.elapsedTime / this.totalDuration) * 100;
        this.elements.sessionProgress.style.width = `${progress}%`;
        
        // Update instruction highlighting
        this.highlightCurrentInstruction(stepIndex);
        
        // Start countdown timer
        this.startCountdownTimer(instruction.duration);
    }

    updateActivityDisplay(instruction, stepIndex) {
        this.elements.currentInstruction.textContent = instruction.text;
        this.elements.timeRemaining.textContent = `Step ${stepIndex + 1} of ${this.currentActivity.instructions.length}`;
        
        // Update animation based on phase
        this.updateAnimation(instruction.phase);
    }

    updateAnimation(phase) {
        const animationIcon = this.elements.animationIcon;
        const animationContainer = this.elements.animationContainer;

        // Reset animation
        animationContainer.style.animation = 'none';
        void animationContainer.offsetWidth; // Trigger reflow

        switch(phase) {
            case 'inhale':
            case 'inhale_left':
            case 'inhale_right':
                animationIcon.className = 'fas fa-arrow-up';
                animationContainer.style.background = 'linear-gradient(135deg, #28a745, #20c997)';
                animationContainer.style.animation = 'breatheIn 4s ease-in-out';
                break;
                
            case 'exhale':
            case 'exhale_left':
            case 'exhale_right':
                animationIcon.className = 'fas fa-arrow-down';
                animationContainer.style.background = 'linear-gradient(135deg, #dc3545, #e83e8c)';
                animationContainer.style.animation = 'breatheOut 4s ease-in-out';
                break;
                
            case 'hold':
                animationIcon.className = 'fas fa-pause';
                animationContainer.style.background = 'linear-gradient(135deg, #ffc107, #fd7e14)';
                animationContainer.style.animation = 'pulse 2s ease-in-out infinite';
                break;
                
            case 'hold_empty':
                animationIcon.className = 'fas fa-stop';
                animationContainer.style.background = 'linear-gradient(135deg, #6c757d, #495057)';
                animationContainer.style.animation = 'pulse 1.5s ease-in-out infinite';
                break;
                
            default:
                animationIcon.className = 'fas fa-wind';
                animationContainer.style.background = 'linear-gradient(135deg, #667eea, #764ba2)';
                animationContainer.style.animation = 'float 3s ease-in-out infinite';
        }
    }

    highlightCurrentInstruction(stepIndex) {
        const instructions = this.elements.instructionsList.querySelectorAll('.instruction-step');
        instructions.forEach((instruction, index) => {
            instruction.classList.toggle('active', index === stepIndex);
            instruction.classList.toggle('completed', index < stepIndex);
        });
    }

    startCountdownTimer(duration) {
        this.timeRemaining = duration;
        
        if (this.timers.countdown) {
            clearInterval(this.timers.countdown);
        }
        
        this.timers.countdown = setInterval(() => {
            if (this.isPaused || !this.isActivityActive) return;
            
            this.timeRemaining--;
            this.elements.timeRemaining.textContent = `${this.timeRemaining}s remaining`;
            
            if (this.timeRemaining <= 0) {
                clearInterval(this.timers.countdown);
            }
        }, 1000);
    }

    speakInstruction(text) {
        if (this.speechSynthesis.speaking) {
            this.speechSynthesis.cancel();
        }

        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = this.settings.speechRate;
        utterance.volume = this.settings.volume;
        utterance.pitch = 1.0;
        
        // Set voice if available
        const selectedVoice = this.availableVoices.find(voice => 
            voice.name.toLowerCase().includes(this.settings.voiceOption.includes('female') ? 'female' : 'male') ||
            voice.name.toLowerCase().includes(this.settings.voiceOption)
        );
        
        if (selectedVoice) {
            utterance.voice = selectedVoice;
        }

        utterance.onend = () => {
            this.currentUtterance = null;
        };

        this.speechSynthesis.speak(utterance);
        this.currentUtterance = utterance;
    }

    pauseActivitySession() {
        if (!this.isActivityActive || this.isPaused) return;

        this.isPaused = true;
        
        // Clear timers
        if (this.timers.activity) {
            clearTimeout(this.timers.activity);
        }
        if (this.timers.countdown) {
            clearInterval(this.timers.countdown);
        }
        
        // Pause speech
        if (this.speechSynthesis.speaking) {
            this.speechSynthesis.pause();
        }
        
        // Pause animations
        this.elements.animationContainer.style.animationPlayState = 'paused';
        
        this.updateControlButtons('paused');
        this.showToast('Activity paused', 'warning');
    }

    resumeActivitySession() {
        if (!this.isActivityActive || !this.isPaused) return;

        this.isPaused = false;
        
        // Resume speech
        if (this.speechSynthesis.paused) {
            this.speechSynthesis.resume();
        }
        
        // Resume animations
        this.elements.animationContainer.style.animationPlayState = 'running';
        
        // Continue activity sequence
        this.runActivitySequence();
        
        this.updateControlButtons('active');
        this.showToast('Activity resumed', 'success');
    }

    stopActivitySession() {
        this.isActivityActive = false;
        this.isPaused = false;
        
        this.cleanupTimers();
        
        if (this.speechSynthesis.speaking) {
            this.speechSynthesis.cancel();
        }
        
        this.resetActivityUI();
        this.showToast('Activity session stopped', 'info');
    }

    async completeActivitySession() {
        if (!this.activitySession) return;

        try {
            const response = await fetch('/api/activities/complete', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: this.activitySession.session_id,
                    activity_id: this.currentActivity.id,
                    duration: this.elapsedTime,
                    rating: 5, // Could be user-rated
                    effectiveness: 'high'
                })
            });

            const data = await response.json();
            this.showToast('Activity completed successfully! 🎉', 'success');
            this.cleanupActivity();
            
            // Show completion message
            this.elements.currentInstruction.textContent = 'Activity completed! Great job!';
            this.elements.animationIcon.className = 'fas fa-check-circle';
            this.elements.animationContainer.style.background = 'linear-gradient(135deg, #28a745, #20c997)';
            this.elements.animationContainer.style.animation = 'celebrate 2s ease-in-out';
            
            // Auto-close modal after delay
            setTimeout(() => {
                const modal = bootstrap.Modal.getInstance(this.elements.activitySessionModal);
                if (modal) {
                    modal.hide();
                }
            }, 3000);

        } catch (error) {
            console.error('Error completing activity:', error);
            this.showToast('Error completing activity', 'error');
        }
    }

    cleanupTimers() {
        Object.values(this.timers).forEach(timer => {
            if (timer) {
                clearTimeout(timer);
                clearInterval(timer);
            }
        });
        this.timers = {
            activity: null,
            countdown: null,
            animation: null
        };
    }

    cleanupActivity() {
        this.cleanupTimers();
        this.isActivityActive = false;
        this.isPaused = false;
        this.currentStep = 0;
        this.elapsedTime = 0;
        this.timeRemaining = 0;
        this.currentActivity = null;
        this.activitySession = null;
        
        if (this.speechSynthesis.speaking) {
            this.speechSynthesis.cancel();
        }
    }

    resetActivityUI() {
        this.updateControlButtons('ready');
        this.elements.sessionProgress.style.width = '0%';
        this.elements.currentInstruction.textContent = 'Get ready to begin...';
        this.elements.timeRemaining.textContent = 'Starting soon...';
        this.elements.animationIcon.className = 'fas fa-wind';
        this.elements.animationContainer.style.background = 'linear-gradient(135deg, #667eea, #764ba2)';
        this.elements.animationContainer.style.animation = 'float 3s ease-in-out infinite';
        
        // Reset instruction highlighting
        const instructions = this.elements.instructionsList.querySelectorAll('.instruction-step');
        instructions.forEach((instruction, index) => {
            instruction.classList.toggle('active', index === 0);
            instruction.classList.remove('completed');
        });
    }

    updateControlButtons(state) {
        const states = {
            ready: {
                start: false,
                pause: true,
                resume: true,
                stop: true
            },
            active: {
                start: true,
                pause: false,
                resume: true,
                stop: false
            },
            paused: {
                start: true,
                pause: true,
                resume: false,
                stop: false
            }
        };

        const buttonState = states[state] || states.ready;
        
        this.elements.startActivityBtn.classList.toggle('d-none', buttonState.start);
        this.elements.pauseActivityBtn.classList.toggle('d-none', buttonState.pause);
        this.elements.resumeActivityBtn.classList.toggle('d-none', buttonState.resume);
        this.elements.stopActivityBtn.classList.toggle('d-none', buttonState.stop);
    }

    // Voice Settings Management
    showVoiceSettings() {
        this.loadVoiceOptions();
        const modal = new bootstrap.Modal(this.elements.voiceSettingsModal);
        modal.show();
    }

    loadVoiceOptions() {
        fetch('/api/activities/voice-options')
            .then(response => response.json())
            .then(voices => {
                this.displayVoiceOptions(voices);
            })
            .catch(error => {
                console.error('Error loading voice options:', error);
            });
    }

    displayVoiceOptions(voices) {
        if (!this.elements.voiceOptions) return;

        this.elements.voiceOptions.innerHTML = voices.map(voice => `
            <div class="voice-option ${this.settings.voiceOption === voice.id ? 'selected' : ''}" 
                 data-voice-id="${voice.id}">
                <div class="text-center">
                    <i class="fas fa-user ${voice.gender === 'female' ? 'text-pink' : 'text-primary'} mb-2"></i>
                    <div class="small fw-bold">${voice.name}</div>
                    <div class="text-muted" style="font-size: 0.7rem;">${voice.description}</div>
                    <div class="text-muted" style="font-size: 0.6rem;">${voice.age_suitability}</div>
                </div>
            </div>
        `).join('');
    }

    selectVoiceOption(voiceId) {
        this.settings.voiceOption = voiceId;
        document.querySelectorAll('.voice-option').forEach(option => {
            option.classList.toggle('selected', option.dataset.voiceId === voiceId);
        });
    }

    updateSpeechRateDisplay() {
        const rate = parseFloat(this.elements.speechRate.value);
        let text = 'Normal Speed';
        if (rate < 1) text = 'Slower';
        if (rate > 1) text = 'Faster';
        this.elements.speechRateValue.textContent = `${text} (${rate}x)`;
    }

    updateVolumeDisplay() {
        const volume = parseFloat(this.elements.volume.value);
        this.elements.volumeValue.textContent = `${Math.round(volume * 100)}% Volume`;
    }

    saveVoiceSettings() {
        this.settings.speechRate = parseFloat(this.elements.speechRate.value);
        this.settings.volume = parseFloat(this.elements.volume.value);
        
        // Save to localStorage
        localStorage.setItem('activitySettings', JSON.stringify(this.settings));
        
        this.showToast('Voice settings saved', 'success');
        bootstrap.Modal.getInstance(this.elements.voiceSettingsModal).hide();
    }

    loadSettings() {
        const saved = localStorage.getItem('activitySettings');
        if (saved) {
            this.settings = { ...this.settings, ...JSON.parse(saved) };
            this.elements.speechRate.value = this.settings.speechRate;
            this.elements.volume.value = this.settings.volume;
            this.updateSpeechRateDisplay();
            this.updateVolumeDisplay();
        }
    }

    loadVoices() {
        // Load available voices
        const loadVoices = () => {
            this.availableVoices = this.speechSynthesis.getVoices();
            console.log(`Loaded ${this.availableVoices.length} voices`);
        };

        this.speechSynthesis.onvoiceschanged = loadVoices;
        loadVoices(); // Initial load
    }

    viewActivityDetails(activityId) {
        // Implementation for detailed activity view
        this.showToast('Detailed activity view coming soon!', 'info');
    }

    showToast(message, type = 'info') {
        if (window.socketUtils && window.socketUtils.showToast) {
            window.socketUtils.showToast(message, type);
        } else {
            // Fallback toast
            const toast = document.createElement('div');
            toast.className = `alert alert-${type} position-fixed top-0 end-0 m-3`;
            toast.style.zIndex = '1060';
            toast.innerHTML = `
                <div class="d-flex">
                    <div>${message}</div>
                    <button type="button" class="btn-close ms-2" data-bs-dismiss="alert"></button>
                </div>
            `;
            document.body.appendChild(toast);
            
            setTimeout(() => {
                toast.remove();
            }, 3000);
        }
    }
}

// CSS Animations for Activities
const activityStyles = `
@keyframes breatheIn {
    0% { transform: scale(0.8); opacity: 0.7; }
    100% { transform: scale(1.2); opacity: 1; }
}

@keyframes breatheOut {
    0% { transform: scale(1.2); opacity: 1; }
    100% { transform: scale(0.8); opacity: 0.7; }
}

@keyframes pulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.05); }
}

@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-10px); }
}

@keyframes celebrate {
    0% { transform: scale(1); }
    50% { transform: scale(1.1); }
    100% { transform: scale(1); }
}

.instruction-step.active {
    background: #e7f3ff !important;
    border-left-color: #007bff !important;
}

.instruction-step.completed {
    opacity: 0.7;
    background: #f8f9fa !important;
}

.phase-inhale { background-color: #28a745; }
.phase-exhale { background-color: #dc3545; }
.phase-hold { background-color: #ffc107; }
.phase-hold-empty { background-color: #6c757d; }
.phase-prepare { background-color: #17a2b8; }

.voice-option.selected {
    border-color: #007bff !important;
    background-color: #e7f3ff !important;
}

.activity-card {
    transition: all 0.3s ease;
}

.activity-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
}
`;

// Inject styles
const styleSheet = document.createElement('style');
styleSheet.textContent = activityStyles;
document.head.appendChild(styleSheet);

// Initialize Activity Manager when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    window.activityManager = new ActivityManager();
});

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ActivityManager;
}