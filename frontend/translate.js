// Google Translate - Custom Language Dropdown Integration

// 1. Sync dropdown with current cookie on page load
(function () {
    var match = document.cookie.match(/googtrans=\/en\/(\w+)/);
    var currentLang = match ? match[1] : 'en';
    var sel = document.getElementById('langSelect');
    if (sel) sel.value = currentLang;
})();

// 2. Change language: set cookie and trigger Google Translate
function changeLanguage(lang) {
    var hostname = window.location.hostname;
    // Do not set domain for localhost to avoid browser cookie rejection
    var domainStr = (hostname === 'localhost' || hostname === '127.0.0.1') ? '' : '; domain=' + hostname;
    var subDomainStr = (hostname === 'localhost' || hostname === '127.0.0.1') ? '' : '; domain=.' + hostname;

    if (lang === 'en') {
        // Clear translation cookies
        document.cookie = 'googtrans=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/';
        if (domainStr) document.cookie = 'googtrans=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/' + domainStr;
        if (subDomainStr) document.cookie = 'googtrans=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/' + subDomainStr;
    } else {
        // Set translation cookies
        document.cookie = 'googtrans=/en/' + lang + '; path=/';
        if (domainStr) document.cookie = 'googtrans=/en/' + lang + '; path=/' + domainStr;
        if (subDomainStr) document.cookie = 'googtrans=/en/' + lang + '; path=/' + subDomainStr;
    }

    // Try programmatic trigger first, fallback to reload
    var combo = document.querySelector('.goog-te-combo');
    if (combo) {
        combo.value = lang === 'en' ? '' : lang;
        combo.dispatchEvent(new Event('change'));
        // Update dropdown display
        var sel = document.getElementById('langSelect');
        if (sel) sel.value = lang;
    } else {
        window.location.reload();
    }
}

// --- CENTRALIZED AUTH & NAVBAR LOGIC ---
document.addEventListener('DOMContentLoaded', function () {
    const authSlot = document.getElementById('auth-slot');
    if (authSlot) {
        const isLoggedIn = localStorage.getItem('isLoggedIn') === 'true';
        const farmerName = localStorage.getItem('farmerName') || 'Farmer';
        
        if (isLoggedIn) {
            authSlot.innerHTML = `
                <div class="dropdown">
                    <button class="btn dropdown-toggle" data-bs-toggle="dropdown" style="border: 2px solid #00b894; color: #00b894; background: transparent; padding: 6px 15px; border-radius: 8px; font-weight: 600;">
                        👤 ${farmerName}
                    </button>
                    <ul class="dropdown-menu dropdown-menu-end shadow border-0 mt-2">
                        <li><a class="dropdown-item py-2" href="profile.html">My Profile</a></li>
                        <li><a class="dropdown-item py-2" href="dashboard.html">Dashboard</a></li>
                        <li><hr class="dropdown-divider"></li>
                        <li><button onclick="handleLogout()" class="dropdown-item py-2 text-danger">Logout</button></li>
                    </ul>
                </div>
            `;
        } else {
            authSlot.innerHTML = `
                <a href="login.html" class="btn" style="background: #00b894; color: white; padding: 8px 20px; border-radius: 12px; font-weight: 600;">Farmer Login</a>
            `;
        }
    }
});

window.handleLogout = function() {
    if (confirm("Sign out of Smart Harvest?")) {
        localStorage.removeItem('isLoggedIn');
        localStorage.removeItem('farmerName');
        localStorage.removeItem('farmerEmail');
        localStorage.removeItem('farmerRegion');
        localStorage.removeItem('farmerSoil');
        window.location.href = "login.html";
    }
};

// 3. Initialize Google Translate in the hidden div
function googleTranslateElementInit() {
    new google.translate.TranslateElement({
        pageLanguage: 'en',
        includedLanguages: 'en,hi,gu',
        layout: google.translate.TranslateElement.InlineLayout.SIMPLE,
        autoDisplay: false
    }, 'google_translate_element');
}

// 4. Dynamically load Google Translate script
(function () {
    var s = document.createElement('script');
    s.type = 'text/javascript';
    s.src = 'https://translate.google.com/translate_a/element.js?cb=googleTranslateElementInit';
    document.body.appendChild(s);
})();
