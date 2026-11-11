/// <reference types="vite/client" />

declare module '*.css' {
  const content: string;
  export default content;
}

declare module 'node:test' {
  export function test(name: string, fn: () => void | Promise<void>): Promise<void>;
  export default test;
}

declare module 'node:assert/strict' {
  const assert: {
    ok(value: unknown, message?: string): asserts value;
    equal<T>(actual: T, expected: T, message?: string): void;
  };
  export default assert;
}
