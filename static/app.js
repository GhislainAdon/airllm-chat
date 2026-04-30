/**
 * AirLLM Chat - Application Logic
 * Handles chat, model management, settings, and system info.
 * Works with the Tailwind-styled HTML template (index.html).
 */
(function () {
  'use strict';

  // ──────────────────────────────────────────────
  // State
  // ──────────────────────────────────────────────
  const state = {
    currentView: 'chat',
    conversations: JSON.parse(localStorage.getItem('airllm-conversations') || '[]'),
    currentConversation: null,
    isGenerating: false,
    engineInfo: null,
    ollamaInfo: null,
    systemInfo: null,
    ollamaModels: [],
    settings: JSON.parse(
      localStorage.getItem('airllm-settings') ||
        '{"temperature":0.7,"maxTokens":2048,"topP":0.9}'
    ),
  };

  // ──────────────────────────────────────────────
  // DOM helpers
  // ──────────────────────────────────────────────
  const $ = (s) => document.querySelector(s);
  const $$ = (s) => document.querySelectorAll(s);

  function escapeHtml(str) {
    const d = document.createElement('div');
    d.textContent = str;
    return d.innerHTML;
  }

  // ──────────────────────────────────────────────
  // Toast notifications
  // ──────────────────────────────────────────────
  function showToast(message, type = 'info') {
    let container = $('#toast-container');
    if (!container) {
      container = document.createElement('div');
      container.id = 'toast-container';
      container.style.cssText =
        'position:fixed;top:16px;right:16px;z-index:9999;display:flex;flex-direction:column;gap:8px;pointer-events:none;';
      document.body.appendChild(container);
    }
    const t = document.createElement('div');
    const colors = {
      info: 'border-l-4 border-indigo-glow bg-navy-700',
      success: 'border-l-4 border-teal-glow bg-navy-700',
      error: 'border-l-4 border-red-400 bg-navy-700',
    };
    t.className = `${colors[type] || colors.info} text-sm text-white/90 rounded-lg px-4 py-3 shadow-lg max-w-sm pointer-events-auto`;
    t.style.animation = 'fadeInToast 0.3s ease';
    t.textContent = message;
    container.appendChild(t);
    setTimeout(() => {
      t.style.opacity = '0';
      t.style.transition = 'opacity 0.3s';
      setTimeout(() => t.remove(), 300);
    }, 3500);
  }

  // Inject toast animation if not present
  if (!document.getElementById('airllm-toast-style')) {
    const s = document.createElement('style');
    s.id = 'airllm-toast-style';
    s.textContent = '@keyframes fadeInToast{from{opacity:0;transform:translateY(-8px)}to{opacity:1;transform:translateY(0)}}';
    document.head.appendChild(s);
  }

  // ──────────────────────────────────────────────
  // API helper
  // ──────────────────────────────────────────────
  async function api(endpoint, options = {}) {
    const resp = await fetch(endpoint, {
      headers: { 'Content-Type': 'application/json', ...options.headers },
      ...options,
    });
    if (!resp.ok) {
      const body = await resp.text().catch(() => '');
      throw new Error(`API ${resp.status}: ${body || resp.statusText}`);
    }
    return resp;
  }

  // ──────────────────────────────────────────────
  // Navigation
  // ──────────────────────────────────────────────
  function switchView(view) {
    state.currentView = view;

    // Delegate to the HTML template's built-in switcher
    if (typeof window._airllmSwitchView === 'function') {
      window._airllmSwitchView(view);
    }

    // Load view-specific data on activation
    if (view === 'models') loadOllamaModels();
    if (view === 'settings') {
      loadSettings();
      loadSystemInfo();
      loadOllamaStatus();
    }
  }

  // ──────────────────────────────────────────────
  // Markdown renderer (no library)
  // ──────────────────────────────────────────────
  function renderMarkdown(text) {
    if (!text) return '';

    let html = escapeHtml(text);

    // Code blocks with language label and copy button
    html = html.replace(/```(\w*)\n?([\s\S]*?)```/g, (_, lang, code) => {
      const langLabel = lang ? `<span class="text-[10px] text-white/30 absolute top-2 left-3 select-none">${escapeHtml(lang)}</span>` : '';
      const raw = code.trim();
      return `<div class="msg-code-block"><button class="copy-code-btn" title="Copy code">Copy</button>${langLabel}<code>${raw}</code></div>`;
    });

    // Inline code (skip inside already-processed code blocks)
    html = html.replace(/`([^`]+)`/g, '<code class="msg-inline-code">$1</code>');

    // Tables
    html = html.replace(
      /^(\|.+\|)\n(\|[\s\-:|]+\|)\n((?:\|.+\|\n?)+)/gm,
      (_, header, separator, body) => {
        const ths = header.split('|').filter(Boolean).map((h) => `<th>${h.trim()}</th>`).join('');
        const rows = body
          .trim()
          .split('\n')
          .map((row) => {
            const tds = row.split('|').filter(Boolean).map((c) => `<td>${c.trim()}</td>`).join('');
            return `<tr>${tds}</tr>`;
          })
          .join('');
        return `<div class="overflow-x-auto my-2"><table><thead><tr>${ths}</tr></thead><tbody>${rows}</tbody></table></div>`;
      }
    );

    // Headings
    html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
    html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
    html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');

    // Horizontal rules
    html = html.replace(/^---$/gm, '<hr class="border-white/10 my-4">');

    // Blockquotes
    html = html.replace(/^&gt; (.+)$/gm, '<blockquote>$1</blockquote>');

    // Bold and italic
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');

    // Links
    html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener" class="text-indigo-glow hover:underline">$1</a>');

    // Unordered lists
    html = html.replace(/^[\-\*] (.+)$/gm, '<li>$1</li>');

    // Ordered lists
    html = html.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');

    // Wrap consecutive <li> in <ul>
    html = html.replace(/((?:<li>.*<\/li>\n?)+)/g, '<ul>$1</ul>');

    // Paragraphs (double newline)
    html = html.replace(/\n\n/g, '</p><p>');

    // Single newlines to <br> (but not inside pre/code)
    html = html.replace(/\n/g, '<br>');

    // Wrap in paragraph if not already wrapped
    if (!html.startsWith('<')) html = `<p>${html}</p>`;

    return html;
  }

  // ──────────────────────────────────────────────
  // Chat: rendering
  // ──────────────────────────────────────────────
  function renderMessage(msg) {
    const isUser = msg.role === 'user';
    const div = document.createElement('div');
    div.className = `flex gap-3 ${isUser ? 'justify-end' : ''} mb-4 animate-[fadeIn_0.3s_ease]`;
    div.style.animation = 'fadeInToast 0.3s ease';

    if (isUser) {
      div.innerHTML = `
        <div class="max-w-2xl">
          <div class="bg-indigo-glow/10 border border-indigo-glow/15 rounded-2xl rounded-tr-sm px-4 py-3">
            <p class="text-sm text-white/90 whitespace-pre-wrap leading-relaxed">${escapeHtml(msg.content)}</p>
          </div>
        </div>
        <div class="w-8 h-8 rounded-full bg-indigo-glow/20 flex items-center justify-center shrink-0 mt-1">
          <span class="material-symbols-outlined text-indigo-glow" style="font-size:16px;">person</span>
        </div>`;
    } else {
      div.innerHTML = `
        <div class="w-8 h-8 rounded-full bg-teal-glow/15 flex items-center justify-center shrink-0 mt-1">
          <span class="text-sm">⚡</span>
        </div>
        <div class="max-w-3xl min-w-0 flex-1">
          <div class="msg-content text-sm text-white/85 leading-relaxed">${renderMarkdown(msg.content)}</div>
        </div>`;
    }
    return div;
  }

  function scrollToBottom() {
    const el = $('#chat-messages');
    if (el) el.scrollTop = el.scrollHeight;
  }

  function showWelcome(show) {
    const w = $('#welcome-screen');
    const chips = $('#chat-chips');
    if (w) w.style.display = show ? '' : 'none';
    if (chips) chips.classList.toggle('hidden', show);
  }

  // ──────────────────────────────────────────────
  // Chat: conversation management
  // ──────────────────────────────────────────────
  function newChat() {
    const conv = {
      id: Date.now().toString(36) + Math.random().toString(36).slice(2, 6),
      title: 'New Chat',
      messages: [],
      created: new Date().toISOString(),
    };
    state.conversations.unshift(conv);
    state.currentConversation = conv;
    saveConversations();
    updateSidebar();
    renderCurrentConversation();
    showWelcome(true);
    if (state.currentView !== 'chat') switchView('chat');
    $('#chat-input')?.focus();
  }

  function loadConversation(id) {
    const conv = state.conversations.find((c) => c.id === id);
    if (!conv) return;
    state.currentConversation = conv;
    updateSidebar();
    renderCurrentConversation();
    if (state.currentView !== 'chat') switchView('chat');
  }

  function deleteConversation(id) {
    state.conversations = state.conversations.filter((c) => c.id !== id);
    if (state.currentConversation?.id === id) {
      state.currentConversation = state.conversations[0] || null;
    }
    saveConversations();
    updateSidebar();
    renderCurrentConversation();
  }

  function saveConversations() {
    try {
      localStorage.setItem('airllm-conversations', JSON.stringify(state.conversations));
    } catch (e) {
      if (e.name === 'QuotaExceededError') {
        state.conversations = state.conversations.slice(0, Math.max(3, state.conversations.length - 5));
        try {
          localStorage.setItem('airllm-conversations', JSON.stringify(state.conversations));
        } catch {
          localStorage.removeItem('airllm-conversations');
        }
        showToast('Old conversations trimmed to free storage.', 'info');
      }
    }
  }

  function updateSidebar() {
    const container = $('#sidebar-chats');
    if (!container) return;
    if (state.conversations.length === 0) {
      container.innerHTML = '<div class="text-center py-6"><p class="text-xs text-white/20">No conversations yet</p></div>';
      return;
    }
    container.innerHTML = state.conversations
      .map(
        (c) => `
      <div class="group flex items-center gap-1 px-3 py-2 rounded-lg cursor-pointer transition-colors
        ${state.currentConversation?.id === c.id ? 'bg-indigo-glow/10 text-white/90' : 'text-white/40 hover:bg-glass-hover hover:text-white/70'}"
        data-conv-id="${c.id}">
        <span class="material-symbols-outlined text-white/20" style="font-size:14px;">chat_bubble_outline</span>
        <span class="flex-1 text-xs truncate">${escapeHtml(c.title)}</span>
        <button class="delete-conv-btn opacity-0 group-hover:opacity-100 transition-opacity text-white/20 hover:text-red-400"
          data-conv-id="${c.id}" title="Delete">
          <span class="material-symbols-outlined" style="font-size:14px;">close</span>
        </button>
      </div>`
      )
      .join('');

    // Bind click events
    container.querySelectorAll('[data-conv-id]').forEach((el) => {
      if (el.classList.contains('delete-conv-btn')) {
        el.addEventListener('click', (e) => {
          e.stopPropagation();
          deleteConversation(el.dataset.convId);
        });
      } else {
        el.addEventListener('click', () => loadConversation(el.dataset.convId));
      }
    });
  }

  function renderCurrentConversation() {
    const container = $('#chat-messages');
    if (!container) return;

    // Remove existing message elements (keep welcome screen)
    container.querySelectorAll('[data-msg-role]').forEach((el) => el.remove());

    if (!state.currentConversation || state.currentConversation.messages.length === 0) {
      showWelcome(true);
      return;
    }
    showWelcome(false);
    state.currentConversation.messages.forEach((msg) => {
      const el = renderMessage(msg);
      el.dataset.msgRole = msg.role;
      container.appendChild(el);
    });
    scrollToBottom();
  }

  // ──────────────────────────────────────────────
  // Chat: sending & streaming
  // ──────────────────────────────────────────────
  function addMessage(role, content) {
    if (!state.currentConversation) newChat();
    const msg = { role, content };
    state.currentConversation.messages.push(msg);

    // Auto-title from first user message
    if (role === 'user' && state.currentConversation.title === 'New Chat') {
      state.currentConversation.title = content.length > 40 ? content.slice(0, 40) + '...' : content;
      updateSidebar();
    }

    const container = $('#chat-messages');
    if (container) {
      const el = renderMessage(msg);
      el.dataset.msgRole = role;
      container.appendChild(el);
      scrollToBottom();
    }
  }

  async function sendMessage() {
    const input = $('#chat-input');
    if (!input) return;
    const text = input.value.trim();
    if (!text || state.isGenerating) return;

    input.value = '';
    input.style.height = 'auto';
    const sendBtn = $('#chat-send');
    if (sendBtn) sendBtn.disabled = true;

    addMessage('user', text);
    saveConversations();

    // Start generating
    state.isGenerating = true;
    toggleStopBtn(true);

    // Create streaming placeholder
    const container = $('#chat-messages');
    const streamDiv = document.createElement('div');
    streamDiv.className = 'flex gap-3 mb-4';
    streamDiv.dataset.msgRole = 'assistant';
    streamDiv.style.animation = 'fadeInToast 0.3s ease';
    streamDiv.innerHTML = `
      <div class="w-8 h-8 rounded-full bg-teal-glow/15 flex items-center justify-center shrink-0 mt-1">
        <span class="text-sm">⚡</span>
      </div>
      <div class="max-w-3xl min-w-0 flex-1">
        <div class="msg-content text-sm text-white/85 leading-relaxed">
          <div class="typing-indicator flex gap-1.5 py-2"><span></span><span></span><span></span></div>
        </div>
      </div>`;
    container.appendChild(streamDiv);
    scrollToBottom();

    const contentEl = streamDiv.querySelector('.msg-content');
    let fullResponse = '';
    let renderTick = 0;

    try {
      const messages = state.currentConversation.messages.map((m) => ({ role: m.role, content: m.content }));
      const resp = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages, stream: true, ...state.settings }),
      });

      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

      const reader = resp.body.getReader();
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
              renderTick++;
              if (renderTick % 3 === 0) {
                contentEl.innerHTML = renderMarkdown(fullResponse);
                scrollToBottom();
              }
            }
          } catch {
            // Skip malformed JSON lines
          }
        }
      }

      // Final render
      if (fullResponse) {
        contentEl.innerHTML = renderMarkdown(fullResponse);
      }
    } catch (err) {
      // Fallback to non-streaming
      console.warn('Streaming failed, trying non-streaming fallback:', err);
      try {
        const messages = state.currentConversation.messages.map((m) => ({ role: m.role, content: m.content }));
        const resp = await api('/api/chat', {
          method: 'POST',
          body: JSON.stringify({ messages, stream: false, ...state.settings }),
        });
        const data = await resp.json();
        fullResponse = data.choices?.[0]?.message?.content || data.content || data.response || '[No response]';
        contentEl.innerHTML = renderMarkdown(fullResponse);
      } catch (err2) {
        contentEl.innerHTML = `<p class="text-red-400 text-sm">Error: ${escapeHtml(err2.message)}</p>`;
        showToast('Failed to get response. Is the model loaded?', 'error');
      }
    }

    scrollToBottom();

    // Finalize
    if (fullResponse) {
      state.currentConversation.messages.push({ role: 'assistant', content: fullResponse });
      saveConversations();
      updateSidebar();
    }

    state.isGenerating = false;
    toggleStopBtn(false);
  }

  function toggleStopBtn(show) {
    const stop = $('#chat-stop');
    const send = $('#chat-send');
    if (stop) {
      stop.style.display = show ? 'flex' : 'none';
    }
    if (send) {
      send.style.display = show ? 'none' : 'flex';
    }
  }

  function stopGeneration() {
    fetch('/api/stop', { method: 'POST' }).catch(() => {});
    state.isGenerating = false;
    toggleStopBtn(false);
    showToast('Generation stopped.', 'info');
  }

  // ──────────────────────────────────────────────
  // Models
  // ──────────────────────────────────────────────
  async function loadOllamaModels() {
    const grid = $('#model-grid');
    const list = $('#models-list');
    if (!grid || !list) return;

    try {
      const resp = await api('/api/ollama/models');
      const data = await resp.json();
      const models = data.models || data || [];

      // Populate grid with cards
      if (models.length === 0) {
        grid.innerHTML = `
          <div class="col-span-full text-center py-12">
            <span class="material-symbols-outlined text-white/10 mb-2 block" style="font-size:48px;">smart_toy</span>
            <p class="text-sm text-white/30">No models found. Pull one from Ollama or add a GGUF path.</p>
          </div>`;
        list.innerHTML = '<p class="text-center py-6 text-xs text-white/20">No local models</p>';
        return;
      }

      grid.innerHTML = models.map((m) => renderModelCard(m)).join('');
      list.innerHTML = models.map((m) => renderModelListItem(m)).join('');

      // Bind card buttons
      bindModelButtons(grid);
      bindModelButtons(list);
    } catch {
      grid.innerHTML = `<div class="col-span-full text-center py-12">
        <p class="text-sm text-white/30">Could not load models. Is Ollama running?</p>
        <button onclick="document.querySelector('#model-refresh-btn')?.click()" class="mt-2 text-xs text-indigo-glow hover:underline">Retry</button>
      </div>`;
      list.innerHTML = '<p class="text-center py-6 text-xs text-red-400/50">Connection failed</p>';
    }
  }

  function renderModelCard(model) {
    const name = model.name || model.model || 'unknown';
    const size = model.size ? formatBytes(model.size) : '';
    const tags = extractTags(name);
    const family = model.details?.family || '';

    return `
      <div class="rounded-xl glass p-5 hover:bg-glass-hover transition-colors group">
        <div class="flex items-start justify-between mb-3">
          <div>
            <h3 class="font-heading font-semibold text-sm text-white">${escapeHtml(name)}</h3>
            ${size ? `<p class="text-xs text-white/40 mt-0.5">${size}</p>` : ''}
          </div>
          <span class="px-2 py-0.5 rounded-full text-[10px] font-medium bg-teal-glow/15 text-teal-glow">Available</span>
        </div>
        <div class="flex gap-1.5 mb-3">
          ${family ? `<span class="px-2 py-0.5 rounded text-[10px] bg-white/5 text-white/40">${escapeHtml(family)}</span>` : ''}
          ${tags.map((t) => `<span class="px-2 py-0.5 rounded text-[10px] bg-white/5 text-white/40">${escapeHtml(t)}</span>`).join('')}
        </div>
        <div class="flex items-center gap-2">
          <button class="load-model-btn flex-1 py-1.5 rounded-lg text-xs font-medium bg-indigo-glow/15 text-indigo-glow hover:bg-indigo-glow/25 transition-colors"
            data-model="${escapeHtml(name)}">Load</button>
          <button class="delete-model-btn py-1.5 px-3 rounded-lg text-xs font-medium text-white/30 hover:text-red-400 hover:bg-red-400/10 transition-colors"
            data-model="${escapeHtml(name)}" title="Delete model">
            <span class="material-symbols-outlined" style="font-size:14px;">delete</span>
          </button>
        </div>
      </div>`;
  }

  function renderModelListItem(model) {
    const name = model.name || model.model || 'unknown';
    const size = model.size ? formatBytes(model.size) : '';

    return `
      <div class="rounded-xl glass p-4 flex items-center justify-between">
        <div class="flex items-center gap-3">
          <div class="w-10 h-10 rounded-lg bg-teal-glow/10 flex items-center justify-center">
            <span class="material-symbols-outlined icon-sm text-teal-glow">memory</span>
          </div>
          <div>
            <p class="text-sm font-medium text-white">${escapeHtml(name)}</p>
            <p class="text-xs text-white/30">${size} · Ollama</p>
          </div>
        </div>
        <div class="flex items-center gap-2">
          <span class="px-2 py-0.5 rounded-full text-[10px] font-medium bg-teal-glow/15 text-teal-glow">Available</span>
          <button class="load-model-btn px-3 py-1.5 rounded-lg text-xs font-medium bg-indigo-glow/15 text-indigo-glow hover:bg-indigo-glow/25 transition-colors"
            data-model="${escapeHtml(name)}">Load</button>
        </div>
      </div>`;
  }

  function bindModelButtons(root) {
    root.querySelectorAll('.load-model-btn').forEach((btn) => {
      btn.addEventListener('click', () => loadModel(btn.dataset.model));
    });
    root.querySelectorAll('.delete-model-btn').forEach((btn) => {
      btn.addEventListener('click', () => deleteModel(btn.dataset.model));
    });
  }

  function extractTags(name) {
    const parts = name.split(/[:/]/);
    return parts.filter((p) => p && p !== parts[0]);
  }

  function formatBytes(bytes) {
    if (typeof bytes === 'string') bytes = parseInt(bytes, 10);
    if (!bytes || isNaN(bytes)) return '';
    const gb = bytes / (1024 * 1024 * 1024);
    return gb >= 1 ? `${gb.toFixed(1)} GB` : `${(bytes / (1024 * 1024)).toFixed(0)} MB`;
  }

  async function loadModel(name) {
    showToast(`Loading ${name}...`, 'info');
    try {
      await api('/api/ollama/load', {
        method: 'POST',
        body: JSON.stringify({ model: name }),
      });
      showToast(`Model "${name}" loaded successfully!`, 'success');
      updateHeaderModel(name);
    } catch (err) {
      showToast(`Failed to load model: ${err.message}`, 'error');
    }
  }

  async function deleteModel(name) {
    if (!confirm(`Delete model "${name}"? This cannot be undone.`)) return;
    try {
      await api('/api/ollama/delete', {
        method: 'POST',
        body: JSON.stringify({ model: name }),
      });
      showToast(`Model "${name}" deleted.`, 'success');
      loadOllamaModels();
    } catch (err) {
      showToast(`Failed to delete model: ${err.message}`, 'error');
    }
  }

  function refreshModels() {
    loadOllamaModels();
    showToast('Refreshing model list...', 'info');
  }

  function searchModels(query) {
    const q = query.toLowerCase();
    $$('#model-grid .rounded-xl, #models-list .rounded-xl').forEach((card) => {
      const text = card.textContent.toLowerCase();
      card.style.display = text.includes(q) ? '' : 'none';
    });
  }

  async function pullModel(name) {
    if (!name || !name.trim()) {
      showToast('Please enter a model name.', 'error');
      return;
    }
    name = name.trim();

    // Close modal
    const modal = $('#pull-model-modal');
    if (modal) modal.classList.remove('open');

    // Show progress banner
    const banner = $('#model-pull-progress');
    const progressName = $('#pull-progress-name');
    const progressPercent = $('#pull-progress-percent');
    const progressBar = $('#pull-progress-bar');
    const progressDetail = $('#pull-progress-detail');
    if (banner) banner.classList.remove('hidden');
    if (progressName) progressName.textContent = `Pulling ${name}...`;
    if (progressPercent) progressPercent.textContent = '0%';
    if (progressBar) progressBar.style.width = '0%';
    if (progressDetail) progressDetail.textContent = 'Starting download...';

    try {
      const resp = await fetch('/api/ollama/pull', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: name }),
      });

      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

      const reader = resp.body.getReader();
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
          if (!trimmed || !trimmed.startsWith('data: ')) continue;

          try {
            const data = JSON.parse(trimmed.slice(6));
            renderPullProgress(data, progressPercent, progressBar, progressDetail);
          } catch {
            // Skip
          }
        }
      }

      showToast(`Model "${name}" pulled successfully!`, 'success');
      loadOllamaModels();
    } catch (err) {
      showToast(`Failed to pull model: ${err.message}`, 'error');
      if (progressDetail) progressDetail.textContent = `Error: ${err.message}`;
    } finally {
      setTimeout(() => {
        if (banner) banner.classList.add('hidden');
      }, 3000);
    }
  }

  function renderPullProgress(data, percentEl, barEl, detailEl) {
    if (data.status === 'downloading' || data.status === 'pulling') {
      const pct = data.total && data.completed ? Math.round((data.completed / data.total) * 100) : 0;
      if (percentEl) percentEl.textContent = `${pct}%`;
      if (barEl) barEl.style.width = `${pct}%`;
      if (detailEl) {
        const completed = formatBytes(data.completed || 0);
        const total = formatBytes(data.total || 0);
        detailEl.textContent = `${completed} / ${total}`;
      }
    } else if (data.status === 'success') {
      if (percentEl) percentEl.textContent = '100%';
      if (barEl) barEl.style.width = '100%';
      if (detailEl) detailEl.textContent = 'Pull complete!';
    } else {
      if (detailEl) detailEl.textContent = data.status || 'Processing...';
    }
  }

  // ──────────────────────────────────────────────
  // Settings
  // ──────────────────────────────────────────────
  function loadSettings() {
    const tempSlider = $('#settings-temp');
    const tempValue = $('#settings-temp-value');
    const maxTokensSlider = $('#settings-max-tokens');
    const maxTokensValue = $('#settings-max-tokens-value');
    const topPSlider = $('#settings-top-p');
    const topPValue = $('#settings-top-p-value');

    if (tempSlider && state.settings.temperature != null) {
      tempSlider.value = state.settings.temperature;
      if (tempValue) tempValue.textContent = parseFloat(state.settings.temperature).toFixed(1);
    }
    if (maxTokensSlider && state.settings.maxTokens != null) {
      maxTokensSlider.value = state.settings.maxTokens;
      if (maxTokensValue) maxTokensValue.textContent = state.settings.maxTokens;
    }
    if (topPSlider && state.settings.topP != null) {
      topPSlider.value = state.settings.topP;
      if (topPValue) topPValue.textContent = parseFloat(state.settings.topP).toFixed(2);
    }
  }

  async function saveSettings() {
    const tempSlider = $('#settings-temp');
    const maxTokensSlider = $('#settings-max-tokens');
    const topPSlider = $('#settings-top-p');

    state.settings = {
      temperature: tempSlider ? parseFloat(tempSlider.value) : 0.7,
      maxTokens: maxTokensSlider ? parseInt(maxTokensSlider.value, 10) : 2048,
      topP: topPSlider ? parseFloat(topPSlider.value) : 0.9,
    };

    // Save locally
    localStorage.setItem('airllm-settings', JSON.stringify(state.settings));

    // Try to push to server
    try {
      await api('/api/settings', {
        method: 'POST',
        body: JSON.stringify(state.settings),
      });
      showToast('Settings saved.', 'success');
    } catch (err) {
      showToast(`Settings saved locally. Server sync failed: ${err.message}`, 'info');
    }
  }

  async function loadSystemInfo() {
    try {
      const resp = await api('/api/system/info');
      const info = await resp.json();
      state.systemInfo = info;
      updateGPUInfo(info);
      updateRAMInfo(info);
    } catch {
      // Silently fail — static placeholder data already in HTML
    }
  }

  function updateGPUInfo(info) {
    const container = $('#system-gpu');
    if (!container || !info.gpu) return;
    const g = info.gpu;
    container.innerHTML = `
      <div class="space-y-4">
        <div>
          <div class="flex items-center justify-between mb-1">
            <span class="text-xs text-white/40">Device</span>
            <span class="text-xs text-white/70 font-mono">${escapeHtml(g.name || g.device || 'N/A')}</span>
          </div>
        </div>
        ${g.vram ? `<div>
          <div class="flex items-center justify-between mb-1.5">
            <span class="text-xs text-white/40">VRAM Usage</span>
            <span class="text-xs text-teal-glow font-mono">${g.vram_used || '?'} / ${g.vram_total || '?'} GB</span>
          </div>
          <div class="w-full h-2 bg-white/5 rounded-full overflow-hidden">
            <div class="h-full bg-gradient-to-r from-teal-glow to-teal-soft rounded-full transition-all" style="width:${g.vram_percent || 0}%"></div>
          </div>
        </div>` : ''}
        ${g.temperature ? `<div>
          <div class="flex items-center justify-between mb-1.5">
            <span class="text-xs text-white/40">Temperature</span>
            <span class="text-xs text-amber-glow font-mono">${g.temperature}°C</span>
          </div>
          <div class="w-full h-2 bg-white/5 rounded-full overflow-hidden">
            <div class="h-full bg-gradient-to-r from-amber-glow to-amber-soft rounded-full transition-all" style="width:${Math.min(100, (g.temperature / 90) * 100)}%"></div>
          </div>
        </div>` : ''}
        ${g.utilization != null ? `<div>
          <div class="flex items-center justify-between">
            <span class="text-xs text-white/40">Utilization</span>
            <span class="text-xs text-indigo-glow font-mono">${g.utilization}%</span>
          </div>
        </div>` : ''}
      </div>`;
  }

  function updateRAMInfo(info) {
    const container = $('#system-ram');
    if (!container) return;
    if (!info.system) return;
    const s = info.system;
    container.innerHTML = `
      <div class="space-y-4">
        ${s.cpu ? `<div>
          <div class="flex items-center justify-between mb-1">
            <span class="text-xs text-white/40">CPU</span>
            <span class="text-xs text-white/70 font-mono">${escapeHtml(s.cpu)}</span>
          </div>
        </div>` : ''}
        ${s.cpu_usage != null ? `<div>
          <div class="flex items-center justify-between mb-1.5">
            <span class="text-xs text-white/40">CPU Usage</span>
            <span class="text-xs text-indigo-glow font-mono">${s.cpu_usage}%</span>
          </div>
          <div class="w-full h-2 bg-white/5 rounded-full overflow-hidden">
            <div class="h-full bg-gradient-to-r from-indigo-glow to-indigo-soft rounded-full transition-all" style="width:${s.cpu_usage}%"></div>
          </div>
        </div>` : ''}
        ${s.ram ? `<div>
          <div class="flex items-center justify-between mb-1.5">
            <span class="text-xs text-white/40">RAM Usage</span>
            <span class="text-xs text-amber-glow font-mono">${s.ram_used || '?'} / ${s.ram_total || '?'} GB</span>
          </div>
          <div class="w-full h-2 bg-white/5 rounded-full overflow-hidden">
            <div class="h-full bg-gradient-to-r from-amber-glow to-amber-soft rounded-full transition-all" style="width:${s.ram_percent || 0}%"></div>
          </div>
        </div>` : ''}
        ${s.platform ? `<div>
          <div class="flex items-center justify-between">
            <span class="text-xs text-white/40">Platform</span>
            <span class="text-xs text-white/70 font-mono">${escapeHtml(s.platform)}</span>
          </div>
        </div>` : ''}
      </div>`;
  }

  async function loadOllamaStatus() {
    const badge = $('#ollama-status-badge');
    if (!badge) return;
    try {
      const resp = await api('/api/ollama/status');
      const data = await resp.json();
      const running = data.running || data.status === 'ok';
      badge.innerHTML = `<span class="w-1.5 h-1.5 rounded-full ${running ? 'bg-teal-glow' : 'bg-red-400'}"></span>${running ? 'Running' : 'Stopped'}`;
      badge.className = `flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[10px] font-medium ${
        running ? 'bg-teal-glow/15 text-teal-glow' : 'bg-red-400/15 text-red-400'
      }`;

      // Update endpoint + path fields if returned
      if (data.endpoint) {
        const ep = $('#ollama-endpoint');
        if (ep) ep.value = data.endpoint;
      }
      if (data.models_path) {
        const mp = $('#ollama-path');
        if (mp) mp.value = data.models_path;
      }
    } catch {
      badge.innerHTML = '<span class="w-1.5 h-1.5 rounded-full bg-red-400"></span>Disconnected';
      badge.className = 'flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[10px] font-medium bg-red-400/15 text-red-400';
    }
  }

  async function testOllamaConnection() {
    const btn = $('#ollama-test-btn');
    if (btn) {
      btn.disabled = true;
      btn.innerHTML = '<span class="material-symbols-outlined animate-spin" style="font-size:14px;">progress_activity</span>Testing...';
    }
    try {
      const resp = await api('/api/ollama/status');
      const data = await resp.json();
      const ok = data.running || data.status === 'ok';
      showToast(ok ? 'Ollama is running and reachable.' : 'Ollama responded but reports not ready.', ok ? 'success' : 'error');
      loadOllamaStatus();
    } catch {
      showToast('Cannot reach Ollama. Check that it is running.', 'error');
      loadOllamaStatus();
    } finally {
      if (btn) {
        btn.disabled = false;
        btn.innerHTML = '<span class="material-symbols-outlined" style="font-size:14px;">wifi_tethering</span>Test Connection';
      }
    }
  }

  function updateHeaderModel(name) {
    const el = $('#header-model-name');
    const badge = $('#header-status-badge');
    if (el) el.textContent = name;
    if (badge) badge.classList.remove('hidden');
    if (badge) badge.classList.add('flex');
  }

  // ──────────────────────────────────────────────
  // Init
  // ──────────────────────────────────────────────
  function init() {
    // Sidebar "New Chat" button
    $$('.sidebar-nav-btn[data-view="chat"]').forEach((btn) => {
      btn.addEventListener('click', () => newChat());
    });

    // Header nav: override to use our switchView which loads data
    $$('.header-nav-btn[data-view]').forEach((btn) => {
      // Remove existing listener by cloning
      const clone = btn.cloneNode(true);
      btn.parentNode.replaceChild(clone, btn);
      clone.addEventListener('click', () => switchView(clone.dataset.view));
    });

    // Sidebar nav: override too
    $$('.sidebar-nav-btn[data-view]').forEach((btn) => {
      const clone = btn.cloneNode(true);
      btn.parentNode.replaceChild(clone, btn);
      const view = clone.dataset.view;
      clone.addEventListener('click', () => {
        if (view === 'chat') newChat();
        else switchView(view);
      });
    });

    // Chat send button
    const sendBtn = $('#chat-send');
    if (sendBtn) sendBtn.addEventListener('click', sendMessage);

    // Chat stop button
    const stopBtn = $('#chat-stop');
    if (stopBtn) stopBtn.addEventListener('click', stopGeneration);

    // Model refresh
    const refreshBtn = $('#model-refresh-btn');
    if (refreshBtn) refreshBtn.addEventListener('click', refreshModels);

    // Model search
    const searchInput = $('#model-search');
    if (searchInput) searchInput.addEventListener('input', () => searchModels(searchInput.value));

    // Pull model button (in modal)
    const pullBtn = $('#model-pull-btn');
    const pullInput = $('#model-pull-input');
    if (pullBtn) {
      pullBtn.addEventListener('click', () => {
        if (pullInput) pullModel(pullInput.value);
      });
    }
    if (pullInput) {
      pullInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && pullBtn) pullBtn.click();
      });
    }

    // Settings save
    const saveSettingsBtn = $('#settings-save-btn');
    if (saveSettingsBtn) saveSettingsBtn.addEventListener('click', saveSettings);

    // Ollama test connection
    const testBtn = $('#ollama-test-btn');
    if (testBtn) testBtn.addEventListener('click', testOllamaConnection);

    // Restore last conversation or show welcome
    if (state.conversations.length > 0) {
      state.currentConversation = state.conversations[0];
      renderCurrentConversation();
    }
    updateSidebar();

    // Initial loads
    loadSettings();
    try { loadSystemInfo(); } catch {}
    try { loadOllamaStatus(); } catch {}
    try { updateHeaderModel(state.settings.activeModel || ''); } catch {}

    // Focus input
    const chatInput = $('#chat-input');
    if (chatInput) chatInput.focus();
  }

  // Boot
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
