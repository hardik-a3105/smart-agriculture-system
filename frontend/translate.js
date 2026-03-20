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
