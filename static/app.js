/**
 * AirLLM Chat - Frontend Application Logic
 * Handles chat UI, API calls, streaming responses, and local storage.
 */

(function() {
    'use strict';

    // --- State ---
    const state = {
        conversations: JSON.parse(localStorage.getItem('airllm_conversations') || '[]'),
        currentConvId: null,
        messages: [],        // { role, content }
        isGenerating: false,
        modelLoaded: false,
        sidebarOpen: true,
    };

    // --- DOM References ---
    const $ = (sel) => document.querySelector(sel);
    const $$ = (sel) => document.querySelectorAll(sel);

    const DOM = {
        sidebar: $('#sidebar'),
        main: $('#main'),
        toggleSidebar: $('#toggle-sidebar'),
        newChatBtn: $('#new-chat-btn'),
        modelSelect: $('#model-select'),
        loadModelBtn: $('#load-model-btn'),
        modelStatus: $('#model-status'),
        tempSlider: $('#temp-slider'),
        tempValue: $('#temp-value'),
        maxTokensInput: $('#max-tokens-input'),
        saveSettingsBtn: $('#save-settings-btn'),
        chatMessages: $('#chat-messages'),
        welcomeScreen: $('#welcome-screen'),
        userInput: $('#user-input'),
        sendBtn: $('#send-btn'),
        stopBtn: $('#stop-btn'),
        clearChatBtn: $('#clear-chat-btn'),
        chatHistory: $('#chat-history'),
        tokenCounter: $('#token-counter'),
        topBarTitle: $('.top-bar-title'),
    };

    // --- API Helper ---
    async function api(endpoint, options = {}) {
        const resp = await fetch(endpoint, {
            headers: { 'Content-Type': 'application/json' },
            ...options,
        });
        return resp.json();
    }

    // --- Toast Notifications ---
    function showToast(message, type = 'info') {
        const container = document.querySelector('.toast-container') || (() => {
            const c = document.createElement('div');
            c.className = 'toast-container';
            document.body.appendChild(c);
            return c;
        })();

        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.textContent = message;
        container.appendChild(toast);
        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transition = 'opacity 0.3s';
            setTimeout(() => toast.remove(), 300);
        }, 3500);
    }

    // --- Model Management ---
    async function loadModels() {
        try {
            const data = await api('/api/models');
            const select = DOM.modelSelect;
            select.innerHTML = '<option value="">-- Select a model --</option>';
            data.models.forEach(m => {
                const opt = document.createElement('option');
                opt.value = m.id;
                opt.textContent = `${m.name} (${m.size})`;
                select.appendChild(opt);
            });
        } catch (e) {
            console.error('Failed to load models list:', e);
        }
    }

    async function checkModelStatus() {
        try {
            const info = await api('/api/info');
            if (info.loaded) {
                state.modelLoaded = true;
                setModelStatus('ready', `Loaded: ${info.model_path.split('/').pop()}`);
            }
        } catch (e) {
            // Server not running yet
        }
    }

    async function loadModel() {
        const modelId = DOM.modelSelect.value;
        if (!modelId) {
            showToast('Please select a model first.', 'error');
            return;
        }

        setModelStatus('loading', 'Loading model...');
        DOM.loadModelBtn.disabled = true;
        DOM.loadModelBtn.textContent = 'Loading...';

        try {
            const result = await api('/api/load', {
                method: 'POST',
                body: JSON.stringify({ model: modelId }),
            });

            if (result.status === 'ok') {
                state.modelLoaded = true;
                setModelStatus('ready', `Loaded: ${modelId.split('/').pop()}`);
                showToast('Model loaded successfully!', 'success');
            } else {
                setModelStatus('error', 'Load failed');
                showToast(result.message, 'error');
            }
        } catch (e) {
            setModelStatus('error', 'Connection error');
            showToast('Failed to connect to server.', 'error');
        }

        DOM.loadModelBtn.disabled = false;
        DOM.loadModelBtn.textContent = 'Load Model';
    }

    function setModelStatus(status, text) {
        const dot = DOM.modelStatus.querySelector('.status-dot');
        dot.className = `status-dot ${status}`;
        DOM.modelStatus.querySelector('span:last-child').textContent = text;
    }

    // --- Settings ---
    async function saveSettings() {
        const temperature = parseFloat(DOM.tempSlider.value);
        const maxTokens = parseInt(DOM.maxTokensInput.value, 10);

        try {
            const result = await api('/api/settings', {
                method: 'POST',
                body: JSON.stringify({
                    temperature: temperature,
                    max_tokens: maxTokens,
                }),
            });
            showToast('Settings saved.', 'success');
        } catch (e) {
            showToast('Failed to save settings.', 'error');
        }
    }

    // --- Chat Functions ---
    function createConversation() {
        const conv = {
            id: Date.now().toString(36) + Math.random().toString(36).slice(2, 6),
            title: 'New Chat',
            messages: [],
            created: new Date().toISOString(),
        };
        state.conversations.unshift(conv);
        state.currentConvId = conv.id;
        state.messages = [];
        saveConversations();
        renderChatHistory();
        renderMessages();
        DOM.topBarTitle.textContent = 'New Chat';
        DOM.welcomeScreen.style.display = '';
    }

    function switchConversation(convId) {
        state.currentConvId = convId;
        const conv = state.conversations.find(c => c.id === convId);
        if (conv) {
            state.messages = [...conv.messages];
            renderMessages();
            renderChatHistory();
            DOM.topBarTitle.textContent = conv.title;
        }
    }

    function saveConversations() {
        const conv = state.conversations.find(c => c.id === state.currentConvId);
        if (conv) {
            conv.messages = [...state.messages];
            // Auto-title from first user message
            if (conv.messages.length > 0 && conv.title === 'New Chat') {
                const firstUser = conv.messages.find(m => m.role === 'user');
                if (firstUser) {
                    conv.title = firstUser.content.slice(0, 40) + (firstUser.content.length > 40 ? '...' : '');
                }
            }
        }
        localStorage.setItem('airllm_conversations', JSON.stringify(state.conversations));
    }

    function renderChatHistory() {
        const container = DOM.chatHistory;
        if (state.conversations.length === 0) {
            container.innerHTML = '<div class="history-empty">No conversations yet</div>';
            return;
        }

        container.innerHTML = state.conversations.map(c => `
            <div class="history-item ${c.id === state.currentConvId ? 'active' : ''}"
                 data-id="${c.id}" title="${escapeHtml(c.title)}">
                ${escapeHtml(c.title)}
            </div>
        `).join('');

        container.querySelectorAll('.history-item').forEach(el => {
            el.addEventListener('click', () => switchConversation(el.dataset.id));
        });
    }

    function escapeHtml(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

    // --- Message Rendering ---
    function renderMessages() {
        if (state.messages.length === 0) {
            DOM.welcomeScreen.style.display = '';
            // Remove all message elements, keep welcome screen
            DOM.chatMessages.querySelectorAll('.message').forEach(el => el.remove());
            return;
        }

        DOM.welcomeScreen.style.display = 'none';
        DOM.chatMessages.innerHTML = '';

        state.messages.forEach(msg => {
            appendMessageToDOM(msg.role, msg.content);
        });

        scrollToBottom();
    }

    function appendMessageToDOM(role, content) {
        const div = document.createElement('div');
        div.className = `message ${role}`;

        const avatarLabel = role === 'user' ? 'U' : 'AI';
        const roleName = role === 'user' ? 'You' : 'AirLLM';

        div.innerHTML = `
            <div class="message-inner">
                <div class="message-avatar">${avatarLabel}</div>
                <div class="message-content">
                    <div class="message-role">${roleName}</div>
                    <div class="message-text">${formatContent(content)}</div>
                </div>
            </div>
        `;

        DOM.chatMessages.appendChild(div);
        return div;
    }

    function createStreamingMessage() {
        const div = document.createElement('div');
        div.className = 'message assistant';
        div.innerHTML = `
            <div class="message-inner">
                <div class="message-avatar">AI</div>
                <div class="message-content">
                    <div class="message-role">AirLLM</div>
                    <div class="message-text">
                        <div class="typing-indicator"><span></span><span></span><span></span></div>
                    </div>
                </div>
            </div>
        `;
        DOM.chatMessages.appendChild(div);
        return div.querySelector('.message-text');
    }

    function formatContent(text) {
        // Basic markdown-like formatting
        let html = escapeHtml(text);

        // Code blocks
        html = html.replace(/```(\w*)\n?([\s\S]*?)```/g, (_, lang, code) => {
            return `<pre><code class="language-${lang}">${code.trim()}</code></pre>`;
        });

        // Inline code
        html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

        // Bold
        html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

        // Italic
        html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');

        // Line breaks
        html = html.replace(/\n/g, '<br>');

        return html;
    }

    function scrollToBottom() {
        DOM.chatMessages.scrollTop = DOM.chatMessages.scrollHeight;
    }

    // --- Send Message ---
    async function sendMessage(text) {
        if (!text.trim() || state.isGenerating) return;

        if (!state.modelLoaded) {
            showToast('Please load a model first!', 'error');
            return;
        }

        const content = text.trim();
        state.messages.push({ role: 'user', content: content });

        // Ensure we have a conversation
        if (!state.currentConvId) {
            createConversation();
        }

        // Hide welcome screen
        DOM.welcomeScreen.style.display = 'none';

        // Add user message to DOM
        appendMessageToDOM('user', content);
        scrollToBottom();

        // Clear input
        DOM.userInput.value = '';
        DOM.userInput.style.height = 'auto';
        updateSendButton();

        // Show generating state
        state.isGenerating = true;
        DOM.sendBtn.style.display = 'none';
        DOM.stopBtn.style.display = 'flex';
        DOM.topBarTitle.textContent = 'Generating...';

        const streamContainer = createStreamingMessage();
        scrollToBottom();

        let fullResponse = '';

        try {
            // Use streaming
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    messages: state.messages,
                    stream: true,
                }),
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';

                for (const line of lines) {
                    const trimmed = line.trim();
                    if (!trimmed || trimmed === 'data: [DONE]') continue;
                    if (!trimmed.startsWith('data: ')) continue;

                    try {
                        const data = JSON.parse(trimmed.slice(6));
                        const token = data.choices?.[0]?.delta?.content || '';
                        if (token) {
                            fullResponse += token;
                            streamContainer.innerHTML = formatContent(fullResponse);
                            scrollToBottom();
                        }
                    } catch (e) {
                        // Skip malformed JSON
                    }
                }
            }

        } catch (e) {
            // If streaming fails, try non-streaming fallback
            console.warn('Streaming failed, trying non-streaming:', e);
            try {
                const data = await api('/api/chat', {
                    method: 'POST',
                    body: JSON.stringify({
                        messages: state.messages,
                        stream: false,
                    }),
                });
                fullResponse = data.choices?.[0]?.message?.content || '[No response]';
                streamContainer.innerHTML = formatContent(fullResponse);
            } catch (e2) {
                fullResponse = `[Error: ${e2.message}]`;
                streamContainer.innerHTML = `<span style="color:var(--error)">${escapeHtml(fullResponse)}</span>`;
            }
        }

        // Finalize
        state.messages.push({ role: 'assistant', content: fullResponse });
        saveConversations();
        renderChatHistory();

        state.isGenerating = false;
        DOM.sendBtn.style.display = 'flex';
        DOM.stopBtn.style.display = 'none';

        // Update title
        const conv = state.conversations.find(c => c.id === state.currentConvId);
        if (conv) {
            DOM.topBarTitle.textContent = conv.title;
        }
    }

    function stopGeneration() {
        // Send stop request
        fetch('/api/stop', { method: 'POST' }).catch(() => {});
        state.isGenerating = false;
        DOM.sendBtn.style.display = 'flex';
        DOM.stopBtn.style.display = 'none';
        DOM.topBarTitle.textContent = 'Stopped';
    }

    // --- UI Helpers ---
    function updateSendButton() {
        DOM.sendBtn.disabled = !DOM.userInput.value.trim();
    }

    function autoResizeTextarea() {
        DOM.userInput.style.height = 'auto';
        DOM.userInput.style.height = Math.min(DOM.userInput.scrollHeight, 200) + 'px';
    }

    function toggleSidebar() {
        state.sidebarOpen = !state.sidebarOpen;
        if (state.sidebarOpen) {
            DOM.sidebar.classList.remove('hidden');
            DOM.main.classList.remove('expanded');
        } else {
            DOM.sidebar.classList.add('hidden');
            DOM.main.classList.add('expanded');
        }
    }

    // --- Event Listeners ---
    function init() {
        // Sidebar toggle
        DOM.toggleSidebar.addEventListener('click', toggleSidebar);

        // New chat
        DOM.newChatBtn.addEventListener('click', createConversation);

        // Clear chat
        DOM.clearChatBtn.addEventListener('click', () => {
            state.messages = [];
            renderMessages();
            showToast('Chat cleared.', 'success');
        });

        // Model loading
        DOM.loadModelBtn.addEventListener('click', loadModel);

        // Temperature slider
        DOM.tempSlider.addEventListener('input', () => {
            DOM.tempValue.textContent = DOM.tempSlider.value;
        });

        // Save settings
        DOM.saveSettingsBtn.addEventListener('click', saveSettings);

        // Send message
        DOM.sendBtn.addEventListener('click', () => {
            sendMessage(DOM.userInput.value);
        });

        // Stop generation
        DOM.stopBtn.addEventListener('click', stopGeneration);

        // Textarea events
        DOM.userInput.addEventListener('input', () => {
            updateSendButton();
            autoResizeTextarea();
        });

        DOM.userInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                if (DOM.userInput.value.trim() && !state.isGenerating) {
                    sendMessage(DOM.userInput.value);
                }
            }
        });

        // Welcome screen tip cards
        $$('.tip-card').forEach(card => {
            card.addEventListener('click', () => {
                const prompt = card.dataset.prompt;
                DOM.userInput.value = prompt;
                updateSendButton();
                sendMessage(prompt);
            });
        });

        // Load saved data
        renderChatHistory();
        if (state.conversations.length > 0) {
            switchConversation(state.conversations[0].id);
        }

        // Initialize
        loadModels();
        checkModelStatus();
        DOM.userInput.focus();

        // Periodically refresh model status
        setInterval(checkModelStatus, 10000);
    }

    // Boot
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

})();
