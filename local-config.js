// API key configuration
// This file reads the API key from api-key.js and manages localStorage

(function () {
	// Get API key from api-key.js (must be loaded before this file)
	const API_KEY = typeof OPENAI_API_KEY !== 'undefined' ? OPENAI_API_KEY : 'sk-your-api-key-here';
	
	// Priority: Use API key from api-key.js if valid, otherwise use localStorage
	// This way, user only needs to fill in api-key.js once, and it will auto-restore even if localStorage is cleared
	const existingKey = localStorage.getItem('openai_api_key');
	const hasValidFileKey = API_KEY && API_KEY !== 'sk-your-api-key-here' && /^sk-/.test(API_KEY) && API_KEY.length > 40;
	const hasValidStoredKey = existingKey && /^sk-/.test(existingKey) && existingKey.length > 40;
	
	if (hasValidFileKey) {
		// File has valid key - use it and save to localStorage (auto-restore)
		try {
			localStorage.setItem('openai_api_key', API_KEY);
			console.log('API key loaded from api-key.js and saved to localStorage');
		} catch (e) {
			console.error('Failed to store API key:', e);
		}
	} else if (hasValidStoredKey) {
		// No valid file key, but localStorage has one - use it
		console.log('API key loaded from localStorage');
	} else {
		// No valid key anywhere
		console.warn('API key not configured. Please edit api-key.js and add your API key.');
	}

	// Manual setter (use in browser console if needed):
	//   window.setOpenAIAPIKey('sk-...')
	window.setOpenAIAPIKey = function (key) {
		if (key && /^sk-/.test(key) && key.length > 40) {
			try { 
				localStorage.setItem('openai_api_key', key);
				console.log('API key updated successfully');
				return true;
			} catch (_) {
				return false;
			}
		}
		console.error('Invalid API key format. Key must start with "sk-" and be at least 40 characters.');
		return false;
	};
})();
