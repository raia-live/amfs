import type { Room, RoomsState } from "./types";

export interface Destination {
  /** null = personal memory */
  id: string | null;
  label: string;
  /** secondary line: where the clip actually lands */
  hint?: string;
}

export const PERSONAL_DESTINATION: Destination = {
  id: null,
  label: "My memory",
  hint: "Private — only your agents",
};

function humanize(segment: string): string {
  return segment
    .replace(/[-_]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/\b\p{Ll}/gu, (c) => c.toUpperCase());
}

/**
 * A room id is a UUID, which means nothing to a user. Prefer the name the API
 * gives us, then the last segment of the entity path, and only fall back to a
 * truncated id so the row is still distinguishable.
 */
export function roomLabel(room: Room): string {
  const name = typeof room.name === "string" ? room.name.trim() : "";
  if (name) return name;
  const path = typeof room.entity_path === "string" ? room.entity_path.trim() : "";
  const segment = path.split("/").filter(Boolean).pop();
  if (segment) return humanize(segment);
  return `Untitled room (${room.room_id.slice(0, 8)})`;
}

export function buildDestinations(roomsState: RoomsState | null): Destination[] {
  const rooms = roomsState?.roomsUnlocked ? roomsState.rooms : [];
  return [
    PERSONAL_DESTINATION,
    ...rooms.map((room) => ({
      id: room.room_id,
      label: roomLabel(room),
      hint:
        typeof room.entity_path === "string" && room.entity_path.trim()
          ? room.entity_path.trim()
          : undefined,
    })),
  ];
}

export function queryTokens(query: string): string[] {
  return query.toLowerCase().split(/\s+/).filter(Boolean);
}

/** Every token has to appear in the label or the memory path (AND, not OR). */
export function filterDestinations(destinations: Destination[], query: string): Destination[] {
  const tokens = queryTokens(query);
  if (tokens.length === 0) return destinations;
  return destinations.filter((d) => {
    const haystack = `${d.label} ${d.hint ?? ""}`.toLowerCase();
    return tokens.every((token) => haystack.includes(token));
  });
}

export interface TextRun {
  text: string;
  match: boolean;
}

/** Split text into matched/unmatched runs so the UI can emphasise what was typed. */
export function highlightRuns(text: string, tokens: string[]): TextRun[] {
  if (tokens.length === 0) return [{ text, match: false }];
  const matched = new Array<boolean>(text.length).fill(false);
  const haystack = text.toLowerCase();
  for (const token of tokens) {
    let from = haystack.indexOf(token);
    while (from !== -1) {
      for (let i = from; i < from + token.length; i++) matched[i] = true;
      from = haystack.indexOf(token, from + 1);
    }
  }

  const runs: TextRun[] = [];
  for (let i = 0; i < text.length; i++) {
    const last = runs[runs.length - 1];
    if (last && last.match === matched[i]) last.text += text[i];
    else runs.push({ text: text[i], match: matched[i] });
  }
  return runs;
}

export function destinationIndex(destinations: Destination[], id: string | null): number {
  const found = destinations.findIndex((d) => d.id === id);
  return found === -1 ? 0 : found;
}
