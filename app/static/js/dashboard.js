document.addEventListener('DOMContentLoaded', function() {
    const socket = io();
    const alertPanel = document.getElementById('alertPanel');
    const noAlertsMessage = document.getElementById('noAlertsMessage');
    const headMovementCount = document.getElementById('headMovementCount');
    const multiplePeopleCount = document.getElementById('multiplePeopleCount');
    const absenceCount = document.getElementById('absenceCount');
    
    // Alert counters
    let alerts = {
        'neck_movement': 0,
        'multiple_people': 0,
        'absence': 0
    };
    
    // Handle incoming alerts
    socket.on('proctor_alert', function(data) {
        if (data.warnings && data.warnings.length > 0) {
            // Hide no alerts message
            if (noAlertsMessage) {
                noAlertsMessage.style.display = 'none';
            }
            
            // Process each warning
            data.warnings.forEach(warning => {
                // Create alert element
                const alertItem = document.createElement('div');
                alertItem.className = `alert-item ${warning.type} alert-pulse`;
                
                // Severity badge
                const severityClass = warning.severity === 'high' ? 'danger' : 
                                   warning.severity === 'medium' ? 'warning' : 'info';
                
                // Update alert counter
                if (warning.type in alerts) {
                    alerts[warning.type]++;
                    
                    // Update counter displays
                    if (headMovementCount) headMovementCount.textContent = alerts['neck_movement'];
                    if (multiplePeopleCount) multiplePeopleCount.textContent = alerts['multiple_people'];
                    if (absenceCount) absenceCount.textContent = alerts['absence'];
                }
                
                alertItem.innerHTML = `
                    <div class="d-flex justify-content-between align-items-center">
                        <strong>${warning.message}</strong>
                        <span class="badge bg-${severityClass}">${warning.severity}</span>
                    </div>
                    <small class="text-muted">${new Date().toLocaleTimeString()}</small>
                    ${warning.duration ? `<div class="progress mt-1" style="height: 4px;">
                        <div class="progress-bar bg-${severityClass}" role="progressbar" style="width: ${Math.min(warning.duration / 10 * 100, 100)}%"></div>
                    </div>` : ''}
                `;
                
                // Add to panel
                if (alertPanel) {
                    alertPanel.insertBefore(alertItem, alertPanel.firstChild);
                    
                    // Remove pulse effect after 5 seconds
                    setTimeout(() => {
                        alertItem.classList.remove('alert-pulse');
                    }, 5000);
                    
                    // Limit number of alerts shown
                    if (alertPanel.children.length > 20) {
                        alertPanel.removeChild(alertPanel.lastChild);
                    }
                }
            });
            
            // Play notification sound
            playAlertSound();
        }
    });
    
    // Function to play alert sound
    function playAlertSound() {
        // Use a simple beep sound created with Web Audio API
        try {
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const oscillator = audioContext.createOscillator();
            const gainNode = audioContext.createGain();
            
            oscillator.type = 'sine';
            oscillator.frequency.value = 800; // Frequency in hertz
            gainNode.gain.value = 0.2; // Volume control
            
            oscillator.connect(gainNode);
            gainNode.connect(audioContext.destination);
            
            oscillator.start();
            
            // Stop the beep after 300ms
            setTimeout(function() {
                oscillator.stop();
            }, 300);
        } catch (e) {
            console.log('Audio alert failed:', e);
        }
    }
    
    // Camera error handling
    const videoFeed = document.querySelector('.video-feed, #video-feed');
    if (videoFeed) {
        videoFeed.onerror = function() {
            const cameraStatus = document.getElementById('cameraStatus');
            if (cameraStatus) {
                cameraStatus.classList.add('warning');
                const badge = cameraStatus.querySelector('.badge');
                if (badge) {
                    badge.className = 'badge bg-danger';
                    badge.textContent = 'Error';
                }
            }
        };
    }
    
    // Keep-alive check for video feed
    setInterval(function() {
        const img = new Image();
        img.onload = function() {
            const cameraStatus = document.getElementById('cameraStatus');
            if (cameraStatus) {
                const badge = cameraStatus.querySelector('.badge');
                if (badge) {
                    badge.className = 'badge bg-success';
                    badge.textContent = 'Active';
                }
            }
        };
        img.onerror = function() {
            const cameraStatus = document.getElementById('cameraStatus');
            if (cameraStatus) {
                const badge = cameraStatus.querySelector('.badge');
                if (badge) {
                    badge.className = 'badge bg-danger';
                    badge.textContent = 'Error';
                }
            }
        };
        
        if (videoFeed) {
            img.src = videoFeed.src + '?' + new Date().getTime();
        }
        
        // Also check camera status endpoint
        fetch('/proctor/camera_status')
            .then(response => {
                if (response.ok) {
                    const cameraStatus = document.getElementById('cameraStatus');
                    if (cameraStatus) {
                        const badge = cameraStatus.querySelector('.badge');
                        if (badge) {
                            badge.className = 'badge bg-success';
                            badge.textContent = 'Active';
                        }
                    }
                }
            })
            .catch(error => {
                console.error('Camera status check failed:', error);
            });
    }, 10000);
    
    // Update statistics from server
    setInterval(function() {
        fetch('/proctor/activity_summary')
            .then(response => response.json())
            .then(data => {
                // Update counter displays using both ID patterns
                const headCount = document.getElementById('head-movement-count') || document.getElementById('headMovementCount');
                const multipleCount = document.getElementById('multiple-people-count') || document.getElementById('multiplePeopleCount');
                const absenceCountEl = document.getElementById('absence-count') || document.getElementById('absenceCount');
                
                if (headCount) headCount.textContent = data.neck_movement_count || 0;
                if (multipleCount) multipleCount.textContent = data.multiple_people_count || 0;
                if (absenceCountEl) absenceCountEl.textContent = data.absence_count || 0;
                
                // Update detection status
                const detectionStatus = document.getElementById('detection-status');
                if (detectionStatus && data.active_warnings) {
                    if (data.active_warnings.neck_movement || 
                        data.active_warnings.multiple_people || 
                        data.active_warnings.absence) {
                        detectionStatus.textContent = 'Warning: Suspicious activity detected';
                        detectionStatus.className = 'text-warning';
                    } else {
                        detectionStatus.textContent = 'Monitoring... All clear';
                        detectionStatus.className = 'text-success';
                    }
                }
            })
            .catch(error => {
                console.error('Failed to fetch activity summary:', error);
            });
    }, 5000);
    
    // Initialize alert panel if it doesn't exist
    if (!alertPanel && document.getElementById('alerts-container')) {
        const alertsContainer = document.getElementById('alerts-container');
        const alertPanel = document.createElement('div');
        alertPanel.id = 'alertPanel';
        alertPanel.className = 'alert-panel';
        alertsContainer.appendChild(alertPanel);
    }
}); 