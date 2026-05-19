**Phase 3: Client Interface**


This subsystem provides the user-facing interface for the Orma application. It intentionally avoids modern frontend frameworks (e.g., React, Vue) and build steps (e.g., Webpack, Vite) in favor of lightweight, standalone HTML/JS files distributed dynamically by the FastAPI backend based on User-Agent sniffing. 

**Architecture & Workflows**


**1. Dual-Interface Delivery**

The frontend architecture relies on structural segregation rather than CSS media queries for primary layout shifts.
* **[Desktop_code.html](./Desktop_code.html)**: Implements a persistent left-hand sidebar for session history and a central chat pane.
* **[Mobile_code.html](./Mobile_code.html)**: Implements a CSS-transitioned off-canvas side drawer (`toggleDrawer()`) to maximize viewport real estate on constrained devices.

**2. Core Dependencies (CDN-Delivered)**

* **Styling:** Tailwind CSS (loaded via CDN with container-queries and forms plugins). Injects a custom Material Design 3 (MD3) dark mode color palette configuration.
* **Markdown Parsing:** `marked.js`.
* **Mathematical Rendering:** `KaTeX` combined with `marked-katex-extension` for seamless LaTeX integration within Markdown.
* **Sanitization:** `DOMPurify` to mitigate Cross-Site Scripting (XSS) vulnerabilities.

**3. Rendering Engine & Stream Processing**

The interface handles Server-Sent Events (SSE) from the Gemini inference backend.
* **Stream Consumption:** Utilizes the Fetch API with `response.body.getReader()` and a `TextDecoder` to process text chunks continuously.
* **LaTeX Stream Sanitization (`streamSanitizer`):** A custom regex/substring function designed to intercept incomplete LaTeX delimiters (e.g., `\\[`, `\\(`) arriving mid-stream. It aggressively truncates the string at the last complete delimiter to prevent KaTeX from crashing or rendering raw source code during inference generation.
* **DOM Injection:** Every streamed chunk is processed through `streamSanitizer` -> `marked.parse` -> `DOMPurify.sanitize` before being injected via `innerHTML`, maintaining strict security during real-time updates.

**4. State Management & Persistence**

* **Local Storage:** Conversation history is maintained entirely on the client side using the browser's `localStorage` API under the key `orma_sessions`.
* **Debounced Syncing:** The `saveCurrentSession()` function implements a 1-second debounce timeout to prevent continuous I/O blocking during active typing or stream rendering.
* **Session Lifecycle:** Automatically purges the active memory array to the last 5 turns (`activeMemory.shift()`) before transmission to strictly control backend context windows.

**5. Network Optimization & Security**

* **Timeout Handling:** Implements an `AbortController` with a hard 60-second timeout on all LLM requests.
* **Rate-Limit Bypassing:** Injects the locally generated `myGhostId` into the `X-Session-ID` HTTP header. This synchronizes with the backend's SlowAPI configuration to prevent IP-based rate limiting from blocking entire school networks sharing a single NAT gateway.
* **Error Handling:** Explicitly traps HTTP 429 (Too Many Requests) and non-200 status codes, mapping them to specific UI warning states without crashing the client execution loop.