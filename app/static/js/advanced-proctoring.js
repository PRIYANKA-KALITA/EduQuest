class AdvancedProctoringSystem {
    constructor(examId, options = {}) {
        this.examId = examId;
        this.options = {
            maxWarnings: 20,
            faceDetectionInterval: 1000,
            violationCooldown: 2000,
            faceDirectionSensitivity: 0.3,
            brightnessSensitivity: 30,
            multiPersonThreshold: 0.8,
            autoSubmitEnabled: true,
            visualFeedback: true,
            soundAlerts: true,
            ...options
        };
        
        this.warningCount = 0;
        this.violations = new Map();
        this.lastViolationTime = new Map();
        this.isMonitoring = false;
        this.socket = null;
        this.video = null;
        this.canvas = null;
        this.ctx = null;
        this.detectionInterval = null;
        
        // Face detection variables
        this.faceDetectionActive = false;
        this.lastFacePosition = null;
        this.consecutiveNoFaceFrames = 0;
        this.consecutiveMultipleFaceFrames = 0;
        this.consecutiveMovementFrames = 0;
        
        // Tab/copy-paste monitoring
        this.isTabActive = true;
        this.lastFocusTime = Date.now();
        this.tabSwitchCount = 0;
        this.copyPasteAttempts = 0;
        
        this.init();
    }
    
    init() {
        console.log('🚀 Advanced Proctoring System initializing...');
        this.initializeSocket();
        this.initializeVideoFeed();
        this.initializeTabMonitoring();
        this.initializeCopyPasteDetection();
        this.initializeKeyboardMonitoring();
        this.startMonitoring();
    }
    
    initializeSocket() {
        try {
            this.socket = io();
            this.socket.on('proctor_alert', (data) => {
                if (data.warnings) {
                    data.warnings.forEach(warning => {
                        this.handleServerAlert(warning);
                    });
                }
            });
            console.log('✅ Socket.IO connected for proctoring');
        } catch (e) {
            console.error('❌ Socket.IO connection failed:', e);
        }
    }
    
    initializeVideoFeed() {
        this.video = document.getElementById('cameraFeed');
        if (!this.video) {
            console.error('❌ Camera feed element not found');
            return;
        }
        
        // Create canvas for face detection
        this.canvas = document.createElement('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.canvas.width = 640;
        this.canvas.height = 480;
        
        // Wait for video to be ready
        this.video.addEventListener('loadedmetadata', () => {
            console.log('📹 Video feed ready for monitoring');
            this.faceDetectionActive = true;
        });
    }
    
    initializeTabMonitoring() {
        // Page visibility change detection
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                this.isTabActive = false;
                this.handleTabSwitch();
            } else {
                this.isTabActive = true;
                this.lastFocusTime = Date.now();
            }
        });
        
        // Window focus/blur events
        window.addEventListener('blur', () => {
            this.isTabActive = false;
            this.handleTabSwitch();
        });
        
        window.addEventListener('focus', () => {
            this.isTabActive = true;
            this.lastFocusTime = Date.now();
        });
        
        // Mouse leave detection
        document.addEventListener('mouseleave', () => {
            if (!this.isTabActive) {
                this.handleTabSwitch();
            }
        });
    }
    
    initializeCopyPasteDetection() {
        // Prevent copy operations
        document.addEventListener('copy', (e) => {
            e.preventDefault();
            this.handleCopyPasteAttempt('copy');
            return false;
        });
        
        // Prevent paste operations
        document.addEventListener('paste', (e) => {
            e.preventDefault();
            this.handleCopyPasteAttempt('paste');
            return false;
        });
        
        // Prevent cut operations
        document.addEventListener('cut', (e) => {
            e.preventDefault();
            this.handleCopyPasteAttempt('cut');
            return false;
        });
        
        // Context menu prevention
        document.addEventListener('contextmenu', (e) => {
            e.preventDefault();
            this.addWarning('Right-click disabled during exam', 'low');
            return false;
        });
    }
    
    initializeKeyboardMonitoring() {
        // Prevent common shortcuts
        document.addEventListener('keydown', (e) => {
            // Ctrl+C, Ctrl+V, Ctrl+X
            if (e.ctrlKey && ['c', 'v', 'x'].includes(e.key.toLowerCase())) {
                e.preventDefault();
                this.handleCopyPasteAttempt(e.key.toLowerCase() === 'c' ? 'copy' : e.key.toLowerCase() === 'v' ? 'paste' : 'cut');
                return false;
            }
            
            // Alt+Tab (tab switching)
            if (e.altKey && e.key === 'Tab') {
                e.preventDefault();
                this.handleTabSwitch();
                return false;
            }
            
            // F12, Ctrl+Shift+I (Developer tools)
            if (e.key === 'F12' || (e.ctrlKey && e.shiftKey && e.key === 'I')) {
                e.preventDefault();
                this.addWarning('Developer tools access blocked', 'high');
                return false;
            }
            
            // Ctrl+U (view source)
            if (e.ctrlKey && e.key === 'u') {
                e.preventDefault();
                this.addWarning('View source blocked', 'medium');
                return false;
            }
            
            // Ctrl+Shift+J (console)
            if (e.ctrlKey && e.shiftKey && e.key === 'J') {
                e.preventDefault();
                this.addWarning('Console access blocked', 'high');
                return false;
            }
        });
    }
    
    startMonitoring() {
        if (this.isMonitoring) return;
        
        this.isMonitoring = true;
        console.log('🔍 Proctoring monitoring started');
        
        // Start face detection
        this.detectionInterval = setInterval(() => {
            this.performFaceDetection();
        }, this.options.faceDetectionInterval);
        
        // Monitor tab activity
        setInterval(() => {
            this.checkTabActivity();
        }, 5000);
    }
    
    performFaceDetection() {
        if (!this.faceDetectionActive || !this.video || this.video.videoWidth === 0) {
            return;
        }
        
        try {
            // Set canvas dimensions to match video
            this.canvas.width = this.video.videoWidth;
            this.canvas.height = this.video.videoHeight;
            
            // Draw current frame
            this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
            
            // Convert to base64 for server processing
            const imageData = this.canvas.toDataURL('image/jpeg', 0.5);
            
            // Send to server for analysis
            this.sendFrameForAnalysis(imageData);
            
        } catch (error) {
            console.error('Face detection error:', error);
        }
    }
    
    sendFrameForAnalysis(imageData) {
        fetch('/proctor/verify_image', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image_data: imageData,
                exam_id: this.examId
            })
        })
        .then(response => response.json())
        .then(data => {
            this.handleDetectionResults(data);
        })
        .catch(error => {
            console.error('Face detection API error:', error);
        });
    }
    
    handleDetectionResults(data) {
        if (data.status === 'warning' && data.alerts) {
            data.alerts.forEach(alert => {
                this.addWarning(alert, 'medium');
            });
        }
        
        // Handle face data
        if (data.face_data) {
            if (data.face_data.face_count === 0) {
                this.consecutiveNoFaceFrames++;
                if (this.consecutiveNoFaceFrames > 10) { // 10 seconds
                    this.addWarning('No face detected - please stay in view', 'high');
                    this.consecutiveNoFaceFrames = 0;
                }
            } else {
                this.consecutiveNoFaceFrames = 0;
            }
            
            if (data.face_data.face_count > 1) {
                this.consecutiveMultipleFaceFrames++;
                if (this.consecutiveMultipleFaceFrames > 5) { // 5 seconds
                    this.addWarning(`${data.face_data.face_count} people detected in frame`, 'high');
                    this.consecutiveMultipleFaceFrames = 0;
                }
            } else {
                this.consecutiveMultipleFaceFrames = 0;
            }
            
            // Handle movement detection
            if (data.face_data.movement && data.face_data.movement.detected) {
                this.consecutiveMovementFrames++;
                if (this.consecutiveMovementFrames > 5) { // 5 seconds
                    this.addWarning('Excessive head movement detected', 'medium');
                    this.consecutiveMovementFrames = 0;
                }
            } else {
                this.consecutiveMovementFrames = 0;
            }
        }
    }
    
    handleTabSwitch() {
        const now = Date.now();
        if (this.shouldTriggerViolation('tab_switch', now)) {
            this.tabSwitchCount++;
            this.addWarning(`Tab switching detected (${this.tabSwitchCount} times)`, 'high');
            this.lastViolationTime.set('tab_switch', now);
        }
    }
    
    handleCopyPasteAttempt(type) {
        const now = Date.now();
        if (this.shouldTriggerViolation(`copy_paste_${type}`, now)) {
            this.copyPasteAttempts++;
            this.addWarning(`${type.toUpperCase()} operation blocked (${this.copyPasteAttempts} attempts)`, 'high');
            this.lastViolationTime.set(`copy_paste_${type}`, now);
        }
    }
    
    checkTabActivity() {
        if (!this.isTabActive) {
            const timeAway = Date.now() - this.lastFocusTime;
            if (timeAway > 10000) { // 10 seconds away
                this.addWarning(`Tab inactive for ${Math.round(timeAway/1000)} seconds`, 'medium');
                this.lastFocusTime = Date.now(); // Reset to prevent spam
            }
        }
    }
    
    shouldTriggerViolation(type, now) {
        const lastTime = this.lastViolationTime.get(type) || 0;
        return (now - lastTime) > this.options.violationCooldown;
    }
    
    addWarning(message, severity = 'medium') {
        this.warningCount++;
        
        console.warn(`⚠️ Warning ${this.warningCount}: ${message}`);
        
        // Update warning display
        this.updateWarningDisplay();
        
        // Play sound alert
        if (this.options.soundAlerts) {
            this.playAlertSound();
        }
        
        // Show visual alert
        this.showVisualAlert(message, severity);
        
        // Emit to server
        if (this.socket) {
            this.socket.emit('exam_warning', {
                exam_id: this.examId,
                message: message,
                severity: severity,
                timestamp: Date.now(),
                warning_count: this.warningCount
            });
        }
        
        // Auto-submit if max warnings reached
        if (this.warningCount >= this.options.maxWarnings && this.options.autoSubmitEnabled) {
            this.autoSubmitExam();
        }
    }
    
    updateWarningDisplay() {
        const warningBadge = document.getElementById('warning-badge');
        const warningCounter = document.getElementById('warningCounter');
        
        if (warningBadge) {
            warningBadge.textContent = `${this.warningCount}/${this.options.maxWarnings}`;
            
            // Change color based on warning count
            warningBadge.className = 'badge ';
            if (this.warningCount === 0) {
                warningBadge.className += 'bg-success';
            } else if (this.warningCount < this.options.maxWarnings * 0.5) {
                warningBadge.className += 'bg-warning';
            } else if (this.warningCount < this.options.maxWarnings * 0.8) {
                warningBadge.className += 'bg-orange';
            } else {
                warningBadge.className += 'bg-danger';
            }
        }
        
        if (warningCounter && this.warningCount > 0) {
            warningCounter.classList.add('animate-pulse');
            setTimeout(() => {
                warningCounter.classList.remove('animate-pulse');
            }, 2000);
        }
    }
    
    showVisualAlert(message, severity) {
        // Create alert element
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${severity === 'high' ? 'danger' : severity === 'medium' ? 'warning' : 'info'} alert-dismissible fade show position-fixed`;
        alertDiv.style.cssText = 'top: 100px; right: 20px; z-index: 9999; max-width: 400px;';
        
        alertDiv.innerHTML = `
            <strong>Proctoring Alert:</strong> ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        document.body.appendChild(alertDiv);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.remove();
            }
        }, 5000);
    }
    
    playAlertSound() {
        try {
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const oscillator = audioContext.createOscillator();
            const gainNode = audioContext.createGain();
            
            oscillator.type = 'sine';
            oscillator.frequency.value = 800;
            gainNode.gain.value = 0.1;
            
            oscillator.connect(gainNode);
            gainNode.connect(audioContext.destination);
            
            oscillator.start();
            setTimeout(() => oscillator.stop(), 200);
        } catch (e) {
            console.log('Audio alert failed:', e);
        }
    }
    
    handleServerAlert(warning) {
        this.addWarning(warning.message, warning.severity);
    }
    
    autoSubmitExam() {
        console.warn('🚨 Auto-submitting exam due to excessive warnings');
        
        // Stop monitoring
        this.stopMonitoring();
        
        // Show final warning
        Swal.fire({
            icon: 'error',
            title: 'Exam Auto-Submitted!',
            html: `
                <div class="text-center">
                    <p><strong>Maximum warnings exceeded (${this.warningCount}/${this.options.maxWarnings})</strong></p>
                    <p>Your exam has been automatically submitted due to proctoring violations.</p>
                </div>
            `,
            allowOutsideClick: false,
            allowEscapeKey: false,
            showConfirmButton: true,
            confirmButtonText: 'View Results'
        }).then(() => {
            // Submit the exam
            const form = document.querySelector('form');
            if (form) {
                form.submit();
            } else {
                window.location.href = '/student/dashboard';
            }
        });
    }
    
    stopMonitoring() {
        this.isMonitoring = false;
        if (this.detectionInterval) {
            clearInterval(this.detectionInterval);
        }
        console.log('🛑 Proctoring monitoring stopped');
    }
    
    // Public methods for external control
    increaseWarning() {
        this.addWarning('Manual warning added', 'medium');
    }
    
    getWarningCount() {
        return this.warningCount;
    }
    
    getViolationSummary() {
        return {
            warnings: this.warningCount,
            tabSwitches: this.tabSwitchCount,
            copyPasteAttempts: this.copyPasteAttempts,
            isMonitoring: this.isMonitoring
        };
    }
} 