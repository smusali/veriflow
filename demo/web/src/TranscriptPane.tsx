import type { ReactNode } from "react";

import type { DisplayTranscript } from "./model";

function formatTime(value: number): string {
  return String(value);
}

export function TranscriptPane({
  transcript,
  highlighted,
  selected,
  onSelect,
  source,
}: {
  transcript: DisplayTranscript;
  highlighted: ReadonlySet<number>;
  selected: number | null;
  onSelect: (index: number | null) => void;
  source: ReactNode;
}) {
  const span = laneSpan(transcript);
  return (
    <section className="pane" aria-label="Transcript">
      <h2>Transcript</h2>
      {transcript.notices.map((notice) => (
        <p key={notice} className="notice">
          {notice}
        </p>
      ))}
      {span === null ? null : (
        <div className="lanes" aria-label="Turn intervals">
          {roles.map((role) => (
            <div key={role} className="lane">
              <span className="lane-name">{role}</span>
              <div className="lane-track">
                {transcript.turns
                  .filter((turn) => turn.role === role)
                  .map((turn) => {
                    const left = ((turn.beginning - span.start) / span.span) * 100;
                    const width = Math.max(((turn.end - turn.beginning) / span.span) * 100, 1.5);
                    return (
                      <button
                        key={turn.index}
                        type="button"
                        className={`lane-bar ${role} ${highlighted.has(turn.index) ? "is-hit" : ""}`}
                        style={{ left: `${left}%`, width: `${width}%` }}
                        onClick={() => {
                          onSelect(selected === turn.index ? null : turn.index);
                        }}
                      >
                        {turn.index}
                      </button>
                    );
                  })}
              </div>
            </div>
          ))}
        </div>
      )}
      {transcript.turns.length === 0 ? null : (
        <ol className="turns">
          {transcript.turns.map((turn) => (
            <li key={turn.index}>
              <button
                type="button"
                className={`turn ${turn.role} ${highlighted.has(turn.index) ? "is-hit" : ""} ${selected === turn.index ? "is-selected" : ""}`}
                onClick={() => {
                  onSelect(selected === turn.index ? null : turn.index);
                }}
              >
                <span className="turn-meta">
                  <span>{turn.index}</span>
                  <span>{turn.role}</span>
                  <span>
                    {formatTime(turn.beginning)}-{formatTime(turn.end)}
                  </span>
                </span>
                {turn.overlaps.length > 0 ? (
                  <span className="overlap">Overlaps turn {turn.overlaps.join(", ")}</span>
                ) : null}
                <span className="turn-text">{turn.content}</span>
              </button>
            </li>
          ))}
        </ol>
      )}
      {source}
    </section>
  );
}

const roles: readonly ("user" | "assistant")[] = ["user", "assistant"];

function laneSpan(transcript: DisplayTranscript): { start: number; span: number } | null {
  if (transcript.turns.length === 0) {
    return null;
  }
  let start = transcript.turns[0]?.beginning ?? 0;
  let end = transcript.turns[0]?.end ?? 0;
  for (const turn of transcript.turns) {
    start = Math.min(start, turn.beginning);
    end = Math.max(end, turn.end);
  }
  return { start, span: Math.max(end - start, 0.001) };
}
