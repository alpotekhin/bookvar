declare module 'node:fs' {
  export function existsSync(path: string): boolean;
  export function mkdirSync(path: string, options?: { recursive?: boolean }): void;
  export function mkdtempSync(prefix: string): string;
  export function readFileSync(path: string): Uint8Array;
  export function readFileSync(path: string, encoding: 'utf8'): string;
  export function realpathSync(path: string): string;
  export function symlinkSync(target: string, path: string): void;
  export function writeFileSync(path: string, data: string): void;
}

declare module 'node:os' {
  export function tmpdir(): string;
}

declare module 'node:crypto' {
  export function createHash(algorithm: string): {
    update(data: Uint8Array): { digest(encoding: 'hex'): string };
  };
}

declare module 'node:path' {
  export function dirname(path: string): string;
  export function isAbsolute(path: string): boolean;
  export function join(...paths: string[]): string;
  export function relative(from: string, to: string): string;
  export function resolve(...paths: string[]): string;
  export const sep: string;
}

declare module 'node:fs/promises' {
  export function access(path: string): Promise<void>;
  export function copyFile(source: string, destination: string): Promise<void>;
  export function mkdir(path: string, options?: { recursive?: boolean }): Promise<void>;
  export function readFile(path: string): Promise<Uint8Array>;
  export function readFile(path: string, encoding: 'utf8'): Promise<string>;
  export function rm(path: string, options?: { recursive?: boolean; force?: boolean }): Promise<void>;
  export function writeFile(path: string, data: string, encoding: 'utf8'): Promise<void>;
}

declare module 'node:url' {
  export function fileURLToPath(url: string): string;
  export function pathToFileURL(path: string): { href: string };
}

interface ImportMeta {
  readonly dirname: string;
  readonly url: string;
}

declare const process: {
  argv: string[];
};
