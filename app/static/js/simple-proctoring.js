class SimpleProctoringSystem {
    constructor(examId, options = {}) {
        this.examId = examId;
        this.options = {
            maxWarnings: 20,
            checkInterval: 3000,  // Check every 3 seconds (much slower)
            ...options
        };
        
        this.warningCount = 0;
        this.isMonitoring = false;
        this.lastCheckTime = 0;
        this.video = null;
        this.canvas = null;
        this.ctx = null;
        this.checkInterval = null;
        
        // Tab monitoring (pure JS)
        this.tabSwitchCount = 0;
        this.copyPasteCount = 0;
        this.isTabActive = true;
        
        this.init();
    }
    
    init() {
        console.log('🚀 Simple Proctoring System initializing...');
        this.initializeVideoFeed();
        this.initializeTabMonitoring();
        this.initializeCopyPasteDetection();
        this.startMonitoring();
    }
    
    initializeVideoFeed() {
        this.video = document.getElementById('cameraFeed');
        if (!this.video) {
            console.error('❌ Camera feed element not found');
            return;
        }
        
        // Create canvas for frame capture
        this.canvas = document.createElement('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.canvas.width = 320;  // Smaller size for better performance
        this.canvas.height = 240;
        
        console.log('📹 Video feed ready for monitoring');
    }
    
    initializeTabMonitoring() {
        // Page visibility change detection
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                this.isTabActive = false;
                this.handleTabSwitch();
            } else {
                this.isTabActive = true;
            }
        });
        
        // Window focus/blur events
        window.addEventListener('blur', () => {
            this.isTabActive = false;
            this.handleTabSwitch();
        });
        
        window.addEventListener('focus', () => {
            this.isTabActive = true;
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
            this.showWarning('Right-click disabled during exam');
            return false;
        });
        
        // Keyboard event monitoring - MODIFIED FOR DEBUGGING
        document.addEventListener('keydown', (e) => {
            const key = e.key.toLowerCase();
            const ctrl = e.ctrlKey;
            const shift = e.shiftKey;
            
            // DEBUGGING: Allow F12 and developer tools shortcuts
            if (key === 'f12' || (ctrl && shift && key === 'i') || (ctrl && shift && key === 'j')) {
                console.log('🔧 DEBUGGING: Developer tools shortcut allowed:', key);
                return; // Allow the shortcut
            }
            
            // Copy/paste detection (still active)
            if (ctrl && (key === 'c' || key === 'v' || key === 'x')) {
                e.preventDefault();
                this.reportViolation('copy_paste', `${key.toUpperCase()} shortcut blocked`);
                this.showWarning(`${key.toUpperCase()} shortcut blocked`);
                return;
            }
            
            // Other restricted shortcuts
            if (ctrl && (key === 'a' || key === 's' || key === 'p' || key === 'f' || key === 'h' || key === 'r' || key === 'n' || key === 't' || key === 'w')) {
                e.preventDefault();
                this.showWarning(`${key.toUpperCase()} shortcut blocked`);
                return;
            }
            
            // Alt+Tab, Windows key, etc.
            if (e.altKey && key === 'tab') {
                e.preventDefault();
                this.reportViolation('alt_tab', 'Alt+Tab blocked');
                this.showWarning('Alt+Tab blocked');
                return;
            }
        });
    }
    
    startMonitoring() {
        if (this.isMonitoring) return;
        
        this.isMonitoring = true;
        console.log('🔍 Simple Proctoring monitoring started');
        
        // Start periodic checking with slower interval
        this.checkInterval = setInterval(() => {
            this.performCheck();
        }, this.options.checkInterval);
    }
    
    performCheck() {
        const now = Date.now();
        
        // Throttle checks to prevent spam
        if (now - this.lastCheckTime < this.options.checkInterval) {
            return;
        }
        
        this.lastCheckTime = now;
        
        if (!this.video || this.video.videoWidth === 0) {
            console.log('📹 Video not ready, skipping check');
            return;
        }
        
        try {
            // Capture frame
            this.canvas.width = this.video.videoWidth;
            this.canvas.height = this.video.videoHeight;
            this.ctx.drawImage(this.video, 0, 0);
            
            // Convert to base64 with lower quality
            const imageData = this.canvas.toDataURL('image/jpeg', 0.3);
            
            // Send single request to backend
            this.sendToBackend(imageData);
            
        } catch (error) {
            console.error('Frame capture error:', error);
        }
    }
    
    sendToBackend(imageData) {
        fetch(`/student/exam/${this.examId}/procrastination-check`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image_data: imageData,
                timestamp: Date.now()
            })
        })
        .then(response => response.json())
        .then(data => {
            this.handleBackendResponse(data);
        })
        .catch(error => {
            console.error('Backend error:', error);
        });
    }
    
    handleBackendResponse(data) {
        if (!data.success) {
            console.error('Backend error:', data.message);
            return;
        }
        
        // Update warning count
        this.warningCount = data.warning_count || 0;
        this.updateWarningDisplay();
        
        // Handle violations
        if (data.violations && data.violations.length > 0) {
            data.violations.forEach(violation => {
                this.showWarning(violation.message);
                console.warn('⚠️ Violation:', violation);
            });
        }
        
        // Auto-submit if limit reached
        if (data.action === 'auto_submit' || this.warningCount >= this.options.maxWarnings) {
            this.autoSubmitExam();
        }
        
        // Log debug info
        if (data.debug) {
            console.log('🔍 Debug info:', data.debug);
        }
    }
    
    handleTabSwitch() {
        this.tabSwitchCount++;
        const message = `Tab switching detected (${this.tabSwitchCount} times)`;
        console.warn('🚨 Tab switch detected:', this.tabSwitchCount);
        
        // Send immediate warning to backend
        this.sendViolationToBackend('tab_switch', message);
    }
    
    handleCopyPasteAttempt(type) {
        this.copyPasteCount++;
        const message = `${type.toUpperCase()} operation blocked (${this.copyPasteCount} attempts)`;
        console.warn('🚨 Copy/paste attempt:', type, this.copyPasteCount);
        
        // Send immediate warning to backend
        this.sendViolationToBackend('copy_paste', message);
    }
    
    sendViolationToBackend(violationType, message) {
        fetch('/student/increment_warning', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                exam_id: this.examId,
                violation_type: violationType,
                message: message,
                verified: true,
                timestamp: Date.now()
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                this.warningCount = data.warning_count;
                this.updateWarningDisplay();
                this.showWarning(message);
                
                console.log('✅ Warning sent to backend:', data.warning_count);
                
                if (data.action === 'auto_submit' || this.warningCount >= this.options.maxWarnings) {
                    this.autoSubmitExam();
                }
            }
        })
        .catch(error => {
            console.error('Failed to send violation:', error);
            // Still show local warning even if backend fails
            this.showWarning(message);
        });
    }
    
    addLocalWarning(message) {
        // These are local warnings that don't go to backend
        // They're already tracked by the backend detection
        this.showWarning(message);
        console.warn('⚠️ Local warning:', message);
    }
    
    updateWarningDisplay() {
        const warningBadge = document.getElementById('warning-badge');
        
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
    }
    
    showWarning(message) {
        // Create alert element
        const alertDiv = document.createElement('div');
        alertDiv.className = 'alert alert-warning alert-dismissible fade show position-fixed';
        alertDiv.style.cssText = 'top: 100px; right: 20px; z-index: 9999; max-width: 400px;';
        
        alertDiv.innerHTML = `
            <strong>Proctoring Alert:</strong> ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        document.body.appendChild(alertDiv);
        
        // Auto-remove after 4 seconds
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.remove();
            }
        }, 4000);
        
        // Play sound
        this.playAlertSound();
    }
    
    playAlertSound() {
        try {
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const oscillator = audioContext.createOscillator();
            const gainNode = audioContext.createGain();
            
            oscillator.type = 'sine';
            oscillator.frequency.value = 600;
            gainNode.gain.value = 0.05;
            
            oscillator.connect(gainNode);
            gainNode.connect(audioContext.destination);
            
            oscillator.start();
            setTimeout(() => oscillator.stop(), 150);
        } catch (e) {
            console.log('Audio alert failed:', e);
        }
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
                // Add hidden input to indicate auto-submission
                const autoSubmitInput = document.createElement('input');
                autoSubmitInput.type = 'hidden';
                autoSubmitInput.name = 'auto_submit';
                autoSubmitInput.value = 'true';
                form.appendChild(autoSubmitInput);
                
                form.submit();
            } else {
                window.location.href = '/student/dashboard';
            }
        });
    }
    
    stopMonitoring() {
        this.isMonitoring = false;
        if (this.checkInterval) {
            clearInterval(this.checkInterval);
        }
        console.log('🛑 Simple Proctoring monitoring stopped');
    }
    
    // Public methods
    getWarningCount() {
        return this.warningCount;
    }
    
    getStatus() {
        return {
            warnings: this.warningCount,
            tabSwitches: this.tabSwitchCount,
            copyPasteAttempts: this.copyPasteCount,
            isMonitoring: this.isMonitoring,
            isTabActive: this.isTabActive
        };
    }
    
    // Developer tools detection - DISABLED FOR DEBUGGING
    checkDevTools = () => {
        // DEBUGGING: Allow developer tools access
        console.log('🔧 Developer tools check - DEBUGGING MODE: Access allowed');
        return; // Exit early to allow dev tools
        
        // Original code commented out for debugging
        /*
        const threshold = 160;
        if (window.outerHeight - window.innerHeight > threshold || 
            window.outerWidth - window.innerWidth > threshold) {
            this.reportViolation('developer_tools', 'Developer tools detected');
        }
        */
    };
} 