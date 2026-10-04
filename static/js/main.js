// Main JavaScript file for VANET Routing Analysis

document.addEventListener('DOMContentLoaded', function() {
    
    // Initialize tooltips
    initializeTooltips();
    
    // Initialize image zoom functionality
    initializeImageZoom();
    
    // Initialize performance metrics animations
    initializeMetricsAnimations();
    
    // Initialize navigation highlighting
    initializeNavigation();
    
    // Initialize loading states
    initializeLoadingStates();
    
    console.log('VANET Analysis application initialized');
});

/**
 * Initialize Bootstrap tooltips
 */
function initializeTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Add tooltips to performance metrics
    const performanceElements = document.querySelectorAll('.badge, .metric-value');
    performanceElements.forEach(element => {
        if (element.textContent.includes('%')) {
            element.setAttribute('data-bs-toggle', 'tooltip');
            element.setAttribute('title', 'Performance metric as percentage');
        }
    });
}

/**
 * Initialize image zoom functionality
 */
function initializeImageZoom() {
    const images = document.querySelectorAll('.card-body img, .plot-image');
    
    images.forEach(img => {
        img.style.cursor = 'pointer';
        img.classList.add('img-zoom');
        
        // Add click handler for full-size view
        img.addEventListener('click', function() {
            openImageModal(this.src, this.alt);
        });
        
        // Add hover effect
        img.addEventListener('mouseenter', function() {
            this.style.transform = 'scale(1.02)';
            this.style.transition = 'transform 0.3s ease';
        });
        
        img.addEventListener('mouseleave', function() {
            this.style.transform = 'scale(1)';
        });
    });
}

/**
 * Open image in modal for better viewing
 */
function openImageModal(src, alt) {
    // Create modal if it doesn't exist
    let modal = document.getElementById('imageModal');
    if (!modal) {
        modal = createImageModal();
    }
    
    const modalImage = modal.querySelector('.modal-body img');
    const modalTitle = modal.querySelector('.modal-title');
    
    modalImage.src = src;
    modalImage.alt = alt;
    modalTitle.textContent = alt || 'Analysis Plot';
    
    const bsModal = new bootstrap.Modal(modal);
    bsModal.show();
}

/**
 * Create image modal dynamically
 */
function createImageModal() {
    const modalHTML = `
        <div class="modal fade" id="imageModal" tabindex="-1">
            <div class="modal-dialog modal-xl">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">Analysis Plot</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body text-center">
                        <img src="" alt="" class="img-fluid">
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                        <button type="button" class="btn btn-primary" onclick="downloadImage()">Download</button>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    document.body.insertAdjacentHTML('beforeend', modalHTML);
    return document.getElementById('imageModal');
}

/**
 * Download current modal image
 */
function downloadImage() {
    const modal = document.getElementById('imageModal');
    const image = modal.querySelector('.modal-body img');
    
    if (image && image.src) {
        const link = document.createElement('a');
        link.href = image.src;
        link.download = image.alt.replace(/\s+/g, '_').toLowerCase() + '.png';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }
}

/**
 * Initialize metrics animations
 */
function initializeMetricsAnimations() {
    const metricCards = document.querySelectorAll('.card[data-metric]');
    
    // Animate metric values on scroll
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                animateMetricCard(entry.target);
            }
        });
    }, { threshold: 0.5 });
    
    metricCards.forEach(card => {
        observer.observe(card);
    });
    
    // Add hover effects to performance tables
    const performanceRows = document.querySelectorAll('.table tbody tr');
    performanceRows.forEach(row => {
        row.addEventListener('mouseenter', function() {
            this.style.backgroundColor = 'rgba(0, 102, 204, 0.05)';
            this.style.transition = 'background-color 0.2s ease';
        });
        
        row.addEventListener('mouseleave', function() {
            this.style.backgroundColor = '';
        });
    });
}

/**
 * Animate metric card values
 */
function animateMetricCard(card) {
    const valueElements = card.querySelectorAll('.metric-value, .display-6');
    
    valueElements.forEach(element => {
        const finalValue = parseFloat(element.textContent);
        if (!isNaN(finalValue)) {
            animateValue(element, 0, finalValue, 1000);
        }
    });
}

/**
 * Animate numeric values
 */
function animateValue(element, start, end, duration) {
    const startTime = performance.now();
    const isPercentage = element.textContent.includes('%');
    const isInteger = Number.isInteger(end);
    
    function updateValue(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        // Easing function
        const easeOutQuart = 1 - Math.pow(1 - progress, 4);
        const currentValue = start + (end - start) * easeOutQuart;
        
        if (isInteger) {
            element.textContent = Math.round(currentValue).toLocaleString();
        } else {
            element.textContent = currentValue.toFixed(2);
        }
        
        if (isPercentage && !element.textContent.includes('%')) {
            element.textContent += '%';
        }
        
        if (progress < 1) {
            requestAnimationFrame(updateValue);
        }
    }
    
    requestAnimationFrame(updateValue);
}

/**
 * Initialize navigation highlighting
 */
function initializeNavigation() {
    const navLinks = document.querySelectorAll('.nav-link');
    const currentPath = window.location.pathname;
    
    navLinks.forEach(link => {
        if (link.getAttribute('href') === currentPath) {
            link.classList.add('active');
        }
        
        // Add smooth transition effects
        link.addEventListener('mouseenter', function() {
            if (!this.classList.contains('active')) {
                this.style.transform = 'translateY(-1px)';
                this.style.transition = 'transform 0.2s ease';
            }
        });
        
        link.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
        });
    });
}

/**
 * Initialize loading states for buttons and forms
 */
function initializeLoadingStates() {
    const buttons = document.querySelectorAll('a.btn, button.btn');
    
    buttons.forEach(button => {
        button.addEventListener('click', function() {
            if (this.href && this.href.includes(window.location.origin)) {
                showLoadingState(this);
            }
        });
    });
}

/**
 * Show loading state for buttons
 */
function showLoadingState(button) {
    const originalText = button.innerHTML;
    const originalDisabled = button.disabled;
    
    button.innerHTML = '<span class="loading"></span> Loading...';
    button.disabled = true;
    
    // Reset after a delay (in case navigation fails)
    setTimeout(() => {
        button.innerHTML = originalText;
        button.disabled = originalDisabled;
    }, 5000);
}

/**
 * Utility function to format numbers
 */
function formatNumber(num, decimals = 2) {
    if (typeof num !== 'number') return num;
    
    if (Math.abs(num) >= 1000000) {
        return (num / 1000000).toFixed(1) + 'M';
    } else if (Math.abs(num) >= 1000) {
        return (num / 1000).toFixed(1) + 'K';
    } else {
        return num.toFixed(decimals);
    }
}

/**
 * Utility function to validate numeric inputs
 */
function validateNumericInput(input) {
    const value = parseFloat(input.value);
    const min = parseFloat(input.min) || -Infinity;
    const max = parseFloat(input.max) || Infinity;
    
    if (isNaN(value) || value < min || value > max) {
        input.classList.add('is-invalid');
        return false;
    } else {
        input.classList.remove('is-invalid');
        input.classList.add('is-valid');
        return true;
    }
}

/**
 * Add smooth scrolling for anchor links
 */
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

/**
 * Performance monitoring
 */
function logPerformance() {
    if ('performance' in window) {
        window.addEventListener('load', () => {
            const loadTime = performance.timing.loadEventEnd - performance.timing.navigationStart;
            console.log(`Page load time: ${loadTime}ms`);
        });
    }
}

// Initialize performance monitoring
logPerformance();

/**
 * Handle responsive table scrolling
 */
function initializeResponsiveTables() {
    const tables = document.querySelectorAll('.table-responsive');
    
    tables.forEach(container => {
        const table = container.querySelector('table');
        if (table && table.scrollWidth > container.clientWidth) {
            container.classList.add('has-scroll');
            
            // Add scroll indicators
            const scrollIndicator = document.createElement('div');
            scrollIndicator.className = 'scroll-indicator';
            scrollIndicator.innerHTML = '<i class="fas fa-arrows-alt-h"></i> Scroll horizontally to view all columns';
            container.insertBefore(scrollIndicator, table);
        }
    });
}

// Initialize responsive tables on load and resize
window.addEventListener('load', initializeResponsiveTables);
window.addEventListener('resize', initializeResponsiveTables);

/**
 * Export functionality for tables and charts
 */
function exportTableToCSV(tableId, filename = 'data.csv') {
    const table = document.getElementById(tableId);
    if (!table) return;
    
    let csv = [];
    const rows = table.querySelectorAll('tr');
    
    for (let i = 0; i < rows.length; i++) {
        const row = [];
        const cols = rows[i].querySelectorAll('td, th');
        
        for (let j = 0; j < cols.length; j++) {
            let cellText = cols[j].innerText.replace(/"/g, '""');
            row.push('"' + cellText + '"');
        }
        
        csv.push(row.join(','));
    }
    
    // Download CSV
    const csvContent = csv.join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
}

// Global function for exporting (can be called from templates)
window.exportTableToCSV = exportTableToCSV;
