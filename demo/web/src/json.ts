export type JsonScalar = string | number | boolean | null;
export type JsonValue = JsonScalar | JsonValue[] | { [key: string]: JsonValue };

export type Read<T> =
  | { readonly ok: true; readonly value: T }
  | { readonly ok: false; readonly message: string };

export function fail<T>(message: string): Read<T> {
  return { ok: false, message };
}

export function decodeJson(text: string, label: string): Read<JsonValue> {
  try {
    return { ok: true, value: normalize(JSON.parse(text)) };
  } catch {
    return fail(`${label} is not valid JSON.`);
  }
}

function normalize(value: JsonValue): JsonValue {
  if (value === null || typeof value === "boolean" || typeof value === "string") {
    return value;
  }
  if (typeof value === "number") {
    return Number.isFinite(value) ? value : null;
  }
  if (Array.isArray(value)) {
    const items: JsonValue[] = [];
    for (const item of value) {
      items.push(normalize(item));
    }
    return items;
  }
  const collected: { [key: string]: JsonValue } = {};
  for (const key of Object.keys(value)) {
    const item = value[key];
    if (item === undefined) {
      continue;
    }
    collected[key] = normalize(item);
  }
  return collected;
}

export function isRecord(value: JsonValue): value is { [key: string]: JsonValue } {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

export function readObject(
  value: JsonValue,
  location: string,
): Read<{ [key: string]: JsonValue }> {
  if (!isRecord(value)) {
    return fail(`${location} must be a JSON object.`);
  }
  return { ok: true, value };
}

export function readArray(value: JsonValue, location: string): Read<JsonValue[]> {
  if (!Array.isArray(value)) {
    return fail(`${location} must be a JSON array.`);
  }
  return { ok: true, value };
}

export function readString(value: JsonValue, location: string): Read<string> {
  if (typeof value !== "string") {
    return fail(`${location} must be a string.`);
  }
  return { ok: true, value };
}

export function readBoolean(value: JsonValue, location: string): Read<boolean> {
  if (typeof value !== "boolean") {
    return fail(`${location} must be a boolean.`);
  }
  return { ok: true, value };
}

export function readNumber(value: JsonValue, location: string): Read<number> {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return fail(`${location} must be a finite number.`);
  }
  return { ok: true, value };
}

export function readInteger(value: JsonValue, location: string): Read<number> {
  const number = readNumber(value, location);
  if (!number.ok) {
    return number;
  }
  if (!Number.isInteger(number.value)) {
    return fail(`${location} must be an integer.`);
  }
  return number;
}

export function readField(
  fields: { [key: string]: JsonValue },
  name: string,
  location: string,
): Read<JsonValue> {
  if (!Object.hasOwn(fields, name)) {
    return fail(`${location}.${name} is required.`);
  }
  const value = fields[name];
  if (value === undefined) {
    return fail(`${location}.${name} is required.`);
  }
  return { ok: true, value };
}

export function readNullableString(value: JsonValue, location: string): Read<string | null> {
  if (value === null) {
    return { ok: true, value: null };
  }
  return readString(value, location);
}

export function readNullableNumber(value: JsonValue, location: string): Read<number | null> {
  if (value === null) {
    return { ok: true, value: null };
  }
  return readNumber(value, location);
}

export function readStrings(value: JsonValue, location: string): Read<string[]> {
  const items = readArray(value, location);
  if (!items.ok) {
    return items;
  }
  const texts: string[] = [];
  for (let index = 0; index < items.value.length; index += 1) {
    const item = items.value[index];
    if (item === undefined) {
      return fail(`${location}[${index}] is missing.`);
    }
    const text = readString(item, `${location}[${index}]`);
    if (!text.ok) {
      return text;
    }
    texts.push(text.value);
  }
  return { ok: true, value: texts };
}

export function readIntegers(value: JsonValue, location: string): Read<number[]> {
  const items = readArray(value, location);
  if (!items.ok) {
    return items;
  }
  const numbers: number[] = [];
  for (let index = 0; index < items.value.length; index += 1) {
    const item = items.value[index];
    if (item === undefined) {
      return fail(`${location}[${index}] is missing.`);
    }
    const number = readInteger(item, `${location}[${index}]`);
    if (!number.ok) {
      return number;
    }
    numbers.push(number.value);
  }
  return { ok: true, value: numbers };
}

export function readMember<T extends string>(
  value: JsonValue,
  location: string,
  allowed: readonly T[],
): Read<T> {
  const text = readString(value, location);
  if (!text.ok) {
    return text;
  }
  for (const item of allowed) {
    if (item === text.value) {
      return { ok: true, value: item };
    }
  }
  return fail(`${location} is unsupported.`);
}
