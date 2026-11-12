/**
 * Main JavaScript file for Online Course Recommendation System.
 * Handles AJAX course bookmarks, dynamic UI updates, rating sliders, and Chart.js initialization.
 */

document.addEventListener("DOMContentLoaded", function () {
  // 1. Auto-dismiss flash alert messages after 5 seconds
  const flashAlerts = document.querySelectorAll(".alert-dismissible");
  flashAlerts.forEach(alert => {
    setTimeout(() => {
      const bsAlert = bootstrap.Alert.getOrCreateInstance(alert);
      if (bsAlert) {
        bsAlert.close();
      }
    }, 5000);
  });

  // 2. Minimum Rating Slider synchronization
  const ratingRange = document.getElementById("min_rating_range");
  const ratingValueDisplay = document.getElementById("min_rating_value");
  if (ratingRange && ratingValueDisplay) {
    ratingRange.addEventListener("input", function () {
      const val = parseFloat(this.value);
      if (val === 0) {
        ratingValueDisplay.textContent = "Any Rating (0.0+)";
      } else {
        ratingValueDisplay.textContent = `${val.toFixed(1)}+ Stars`;
      }
    });
  }

  // 3. AJAX Bookmark Toggle functionality
  const bookmarkButtons = document.querySelectorAll(".ajax-bookmark");
  bookmarkButtons.forEach(button => {
    button.addEventListener("click", function (e) {
      e.preventDefault();
      const courseId = this.getAttribute("data-course-id");
      const isSaved = this.classList.contains("active");
      const endpoint = isSaved ? `/unsave-course/${courseId}` : `/save-course/${courseId}`;

      fetch(endpoint, {
        method: "POST",
        headers: {
          "X-Requested-With": "XMLHttpRequest",
          "Content-Type": "application/json"
        }
      })
      .then(response => {
        if (response.redirected) {
          window.location.href = response.url;
          return null;
        }
        return response.json();
      })
      .then(data => {
        if (!data) return;

        if (data.status === "success" || data.status === "exists") {
          if (data.is_saved) {
            this.classList.add("active");
            this.innerHTML = '<i class="bi bi-bookmark-fill me-1"></i> Saved';
            showToast("Success", data.message, "success");
          } else {
            this.classList.remove("active");
            this.innerHTML = '<i class="bi bi-bookmark me-1"></i> Save Course';
            showToast("Info", data.message, "info");

            // If we are currently on the dashboard, remove the card smoothly
            const courseCardCol = this.closest(".dashboard-course-item");
            if (courseCardCol) {
              courseCardCol.style.transition = "opacity 0.3s, transform 0.3s";
              courseCardCol.style.opacity = "0";
              courseCardCol.style.transform = "scale(0.95)";
              setTimeout(() => {
                courseCardCol.remove();
                checkEmptyDashboard();
              }, 300);
            }
          }
        } else {
          showToast("Notice", data.message || "Action could not be completed.", "warning");
        }
      })
      .catch(error => {
        console.error("Bookmark request failed:", error);
      });
    });
  });

  // Helper function to check if dashboard saved list is empty
  function checkEmptyDashboard() {
    const remaining = document.querySelectorAll(".dashboard-course-item");
    const container = document.getElementById("saved-courses-container");
    const emptyState = document.getElementById("empty-saved-state");
    if (remaining.length === 0 && emptyState && container) {
      emptyState.classList.remove("d-none");
    }
  }

  // 4. Dynamic Toast Notifications
  function showToast(title, message, type = "info") {
    let toastContainer = document.getElementById("toast-container");
    if (!toastContainer) {
      toastContainer = document.createElement("div");
      toastContainer.id = "toast-container";
      toastContainer.className = "toast-container position-fixed bottom-0 end-0 p-3";
      document.body.appendChild(toastContainer);
    }

    const toastId = "toast-" + Date.now();
    const bgClass = type === "success" ? "text-bg-success" : type === "danger" ? "text-bg-danger" : "text-bg-primary";

    const toastHtml = `
      <div id="${toastId}" class="toast align-items-center ${bgClass} border-0" role="alert" aria-live="assertive" aria-atomic="true">
        <div class="d-flex">
          <div class="toast-body">
            <strong>${title}:</strong> ${message}
          </div>
          <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
        </div>
      </div>
    `;

    toastContainer.insertAdjacentHTML("beforeend", toastHtml);
    const toastElement = document.getElementById(toastId);
    const toast = new bootstrap.Toast(toastElement, { delay: 3500 });
    toast.show();

    toastElement.addEventListener("hidden.bs.toast", () => {
      toastElement.remove();
    });
  }

  // 5. Select All / Deselect All interest helper
  const selectAllInterestsBtn = document.getElementById("select-all-interests");
  if (selectAllInterestsBtn) {
    selectAllInterestsBtn.addEventListener("click", function (e) {
      e.preventDefault();
      const checkboxes = document.querySelectorAll("input[name='interests']");
      const allChecked = Array.from(checkboxes).every(cb => cb.checked);
      checkboxes.forEach(cb => cb.checked = !allChecked);
      this.textContent = allChecked ? "Select All" : "Clear All";
    });
  }
});
