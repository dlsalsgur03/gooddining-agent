const messagesEl = document.getElementById("messages");
const formEl = document.getElementById("chat-form");
const inputEl = document.getElementById("chat-input");

function getOrCreateId(key) {
  let value = localStorage.getItem(key);
  if (!value) {
    value = crypto.randomUUID();
    localStorage.setItem(key, value);
  }
  return value;
}

const userId = getOrCreateId("gooddining_user_id");
const sessionId = getOrCreateId("gooddining_session_id");

function appendMessage(text, role) {
  const el = document.createElement("div");
  el.className = `message ${role}`;
  el.textContent = text;
  messagesEl.appendChild(el);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return el;
}

formEl.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = inputEl.value.trim();
  if (!text) return;

  appendMessage(text, "user");
  inputEl.value = "";
  inputEl.disabled = true;

  const pending = appendMessage("생각 중...", "agent pending");

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: userId, session_id: sessionId, message: text }),
    });
    if (!response.ok) {
      throw new Error(`서버 오류 (${response.status})`);
    }
    const data = await response.json();
    pending.textContent = data.reply;
    pending.classList.remove("pending");
  } catch (err) {
    pending.textContent = `오류가 발생했어요: ${err.message}`;
    pending.classList.remove("pending");
  } finally {
    inputEl.disabled = false;
    inputEl.focus();
  }
});
