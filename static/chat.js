const messagesEl = document.getElementById("messages");
const formEl = document.getElementById("chat-form");
const inputEl = document.getElementById("chat-input");
const panelTitleEl = document.getElementById("panel-title");
const panelContentEl = document.getElementById("panel-content");
const calMonthLabelEl = document.getElementById("cal-month-label");
const calGridEl = document.getElementById("calendar-grid");
const profileViewBtn = document.getElementById("profile-view-btn");
const profileModalEl = document.getElementById("profile-modal");
const profileModalContentEl = document.getElementById("profile-modal-content");
const profileModalCloseBtn = document.getElementById("profile-modal-close");

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

let calendarCursor = new Date();
let markedDates = new Set();
let selectedDate = null;

function formatDate(d) {
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

const todayStr = formatDate(new Date());
selectedDate = todayStr;

function appendMessage(text, role) {
  const el = document.createElement("div");
  el.className = `message ${role}`;
  el.textContent = text;
  messagesEl.appendChild(el);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return el;
}

// ---------- 내 정보 ----------

const GENDER_LABELS = { male: "남성", female: "여성" };
const ACTIVITY_LABELS = {
  sedentary: "매우 적음",
  light: "적음",
  moderate: "보통",
  active: "많음",
  very_active: "매우 많음",
};
const GOAL_LABELS = { lose: "감량", maintain: "유지", gain: "증량" };

async function openProfileModal() {
  profileModalEl.hidden = false;
  profileModalContentEl.textContent = "불러오는 중...";
  try {
    const response = await fetch(`/profile/${userId}`);
    if (response.status === 404) {
      profileModalContentEl.textContent = "아직 입력된 정보가 없어요. 채팅으로 알려주세요!";
      return;
    }
    if (!response.ok) {
      throw new Error(`서버 오류 (${response.status})`);
    }
    const profile = await response.json();
    profileModalContentEl.innerHTML =
      `${GENDER_LABELS[profile.gender]}, ${profile.age}세<br>` +
      `${profile.height_cm}cm / ${profile.weight_kg}kg<br>` +
      `활동량: ${ACTIVITY_LABELS[profile.activity_level]} · 목표: ${GOAL_LABELS[profile.goal]}` +
      (profile.allergies.length ? `<br>알러지: ${profile.allergies.join(", ")}` : "") +
      (profile.disliked_ingredients.length
        ? `<br>비선호 재료: ${profile.disliked_ingredients.join(", ")}`
        : "");
  } catch (err) {
    profileModalContentEl.textContent = `불러오지 못했어요: ${err.message}`;
  }
}

function closeProfileModal() {
  profileModalEl.hidden = true;
}

profileViewBtn.addEventListener("click", openProfileModal);
profileModalCloseBtn.addEventListener("click", closeProfileModal);
profileModalEl.addEventListener("click", (event) => {
  if (event.target === profileModalEl) closeProfileModal();
});

// ---------- 식단 달력/사이드 패널 ----------

function renderMealPlanHtml(mealPlan) {
  const macros = mealPlan.daily_macros;
  let html =
    `<div class="plan-summary">목표 ${mealPlan.daily_calorie_target.toFixed(0)}kcal<br>` +
    `단백질 ${macros.protein_g.toFixed(0)}g / 탄수화물 ${macros.carbs_g.toFixed(0)}g / ` +
    `지방 ${macros.fat_g.toFixed(0)}g</div>`;

  for (const meal of mealPlan.meals) {
    html += `<div class="plan-meal"><strong>${meal.meal_type}</strong><ul>`;
    for (const dish of meal.dishes) {
      const label = dish.brand ? `${dish.name} (${dish.brand})` : dish.name;
      html += `<li>${label} - ${dish.calories.toFixed(0)}kcal</li>`;
    }
    html += "</ul></div>";
  }
  return html;
}

async function loadMealForDate(dateStr, title) {
  selectedDate = dateStr;
  renderCalendar();

  panelTitleEl.textContent = title;
  panelContentEl.textContent = "불러오는 중...";
  try {
    const response = await fetch(`/meals/${userId}/${dateStr}`);
    if (response.status === 404) {
      panelContentEl.textContent = "이 날짜에 저장된 식단이 없어요.";
      return;
    }
    if (!response.ok) {
      throw new Error(`서버 오류 (${response.status})`);
    }
    const mealPlan = await response.json();
    panelContentEl.innerHTML = renderMealPlanHtml(mealPlan);
  } catch (err) {
    panelContentEl.textContent = `불러오지 못했어요: ${err.message}`;
  }
}

async function loadMarkedDates() {
  try {
    const response = await fetch(`/meals/${userId}/dates`);
    const data = await response.json();
    markedDates = new Set(data.dates);
  } catch (err) {
    markedDates = new Set();
  }
  renderCalendar();
}

function renderCalendar() {
  const year = calendarCursor.getFullYear();
  const month = calendarCursor.getMonth();
  calMonthLabelEl.textContent = `${year}년 ${month + 1}월`;
  calGridEl.innerHTML = "";

  const firstWeekday = new Date(year, month, 1).getDay();
  const daysInMonth = new Date(year, month + 1, 0).getDate();

  for (let i = 0; i < firstWeekday; i++) {
    calGridEl.appendChild(document.createElement("div"));
  }

  for (let day = 1; day <= daysInMonth; day++) {
    const dateStr = formatDate(new Date(year, month, day));
    const cell = document.createElement("button");
    cell.type = "button";
    cell.className = "cal-day";
    cell.textContent = String(day);
    if (markedDates.has(dateStr)) cell.classList.add("has-data");
    if (dateStr === todayStr) cell.classList.add("is-today");
    if (dateStr === selectedDate) cell.classList.add("is-selected");
    cell.addEventListener("click", () => loadMealForDate(dateStr, dateStr));
    calGridEl.appendChild(cell);
  }
}

document.getElementById("cal-prev").addEventListener("click", () => {
  calendarCursor = new Date(calendarCursor.getFullYear(), calendarCursor.getMonth() - 1, 1);
  renderCalendar();
});

document.getElementById("cal-next").addEventListener("click", () => {
  calendarCursor = new Date(calendarCursor.getFullYear(), calendarCursor.getMonth() + 1, 1);
  renderCalendar();
});

// ---------- 채팅 ----------

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

    if (data.meal_plan) {
      loadMealForDate(todayStr, "오늘의 식단");
      loadMarkedDates();
    }
  } catch (err) {
    pending.textContent = `오류가 발생했어요: ${err.message}`;
    pending.classList.remove("pending");
  } finally {
    inputEl.disabled = false;
    inputEl.focus();
  }
});

loadMealForDate(todayStr, "오늘의 식단");
loadMarkedDates();
