import { Readability } from "@mozilla/readability";
import { defineUnlistedScript } from "wxt/utils/define-unlisted-script";
import type { ExtractResult } from "@/utils/types";

/**
 * Injected into the active tab via scripting.executeScript ONLY on explicit
 * user action (popup save, keyboard shortcut, context menu). The return
 * value becomes InjectionResult.result in the background worker.
 */
export default defineUnlistedScript((): ExtractResult => {
  const selection = window.getSelection()?.toString().trim() || undefined;

  let title = document.title || location.href;
  let text = "";
  let byline: string | undefined;
  let excerpt: string | undefined;

  try {
    // Readability mutates its input, so parse a clone.
    const clone = document.cloneNode(true) as Document;
    const article = new Readability(clone).parse();
    if (article) {
      title = article.title || title;
      text = (article.textContent ?? "").trim();
      byline = article.byline ?? undefined;
      excerpt = article.excerpt ?? undefined;
    }
  } catch {
    // fall through to body text
  }

  if (!text) {
    text = (document.body?.innerText ?? "").trim();
  }

  return { title, text, byline, excerpt, selection };
});
