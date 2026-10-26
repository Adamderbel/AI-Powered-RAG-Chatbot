const form = document.querySelector('#chat-form');
const input = document.querySelector('#question');
const messages = document.querySelector('#messages');
const welcome = document.querySelector('#welcome');
const status = document.querySelector('#status');
const send = document.querySelector('#send');
const newChat = document.querySelector('#new-chat');
let sessionId = null;
let busy = false;

function addMessage(role, text, extraClass = '') {
  const message = document.createElement('article');
  message.className = `message ${role} ${extraClass}`;
  const label = document.createElement('div');
  label.className = 'label';
  label.textContent = role === 'user' ? 'You' : 'Knowledge Base';
  const body = document.createElement('div');
  body.className = 'body';
  body.textContent = text;
  message.append(label, body);
  messages.append(message);
  messages.scrollTop = messages.scrollHeight;
  return message;
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const question = input.value.trim();
  if (!question || busy) return;
  busy = true;
  send.disabled = true;
  newChat.disabled = true;
  input.disabled = true;
  welcome.hidden = true;
  addMessage('user', question);
  input.value = '';
  const pending = addMessage('assistant', 'Thinking…', 'pending');
  status.textContent = 'Generating an answer. The first response may take a little longer.';
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 180000);
  try {
    const response = await fetch('/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, session_id: sessionId }),
      signal: controller.signal,
    });
    if (!response.ok) throw new Error(`The server returned an error (${response.status}). Please try again.`);
    const data = await response.json();
    if (typeof data.answer !== 'string' || typeof data.session_id !== 'string') {
      throw new Error('The server returned an unexpected response. Please try again.');
    }
    sessionId = data.session_id;
    pending.remove();
    addMessage('assistant', data.answer);
    status.textContent = 'Ready for your next question.';
  } catch (error) {
    pending.remove();
    let message = error.message;
    if (error.name === 'AbortError') message = 'The response took too long. Check that Ollama is running, then try again.';
    else if (error instanceof TypeError) message = 'Cannot reach the backend. Check that the app is running, then try again.';
    addMessage('assistant', message, 'error');
    input.value = question;
    status.textContent = 'Your question is ready to retry.';
  } finally {
    clearTimeout(timeout);
    busy = false;
    send.disabled = false;
    newChat.disabled = false;
    input.disabled = false;
    input.focus();
  }
});

input.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    form.requestSubmit();
  }
});

document.querySelectorAll('[data-question]').forEach((button) => {
  button.addEventListener('click', () => {
    input.value = button.dataset.question;
    form.requestSubmit();
  });
});

newChat.addEventListener('click', () => {
  if (busy) return;
  sessionId = null;
  messages.querySelectorAll('.message').forEach((message) => message.remove());
  welcome.hidden = false;
  input.value = '';
  status.textContent = 'New conversation started.';
  input.focus();
});
