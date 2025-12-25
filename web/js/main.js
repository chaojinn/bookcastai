(() => {
  async function refreshSession() {
    console.log("refreshSession called");
    try {
      const res = await fetch('/auth/session/refresh', {
        method: 'POST',
        credentials: 'include'
      });
      return res.ok;
    } catch (err) {
      console.warn('Failed to refresh session', err);
      return false;
    }
  }

  function withCredentials(options) {
    const next = options ? { ...options } : {};
    if (!next.credentials) next.credentials = 'include';
    if (next.headers) next.headers = { ...next.headers };
    return next;
  }

  async function fetchWithRefresh(url, options) {
    const firstOptions = withCredentials(options);
    const res = await fetch(url, firstOptions);
    if (res.status !== 401) return res;
    const refreshed = await refreshSession();
    if (!refreshed) return res;
    return fetch(url, withCredentials(options));
  }

  window.refreshSession = refreshSession;
  window.fetchWithRefresh = fetchWithRefresh;
})();
