/**
 * Gemini Agent Service
 * 使用 Gemini CLI 子进程调用暹罗猫 (Gemini)
 *
 * 双 Adapter 架构:
 *   gemini-cli (默认):  spawn 'gemini' CLI + NDJSON → 全自动 headless
 *   antigravity (opt-in): spawn Antigravity IDE → MCP 回传 → 半自动
 *
 * gemini CLI NDJSON 事件格式 (v0.27.2):
 *   init              → session_init (含 session_id)
 *   message/assistant  → text (content 字段)
 *   tool_use           → tool_use
 *   tool_result        → 跳过
 *   message/user       → 跳过 (echo)
 *   result/success     → 跳过
 *   result/error       → error
 */

import { spawn as nodeSpawn } from 'node:child_process';
import { createHash, randomUUID } from 'node:crypto';
import { existsSync, mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { homedir, tmpdir } from 'node:os';
import { basename, extname, join } from 'node:path';
import { type CatId, createCatId } from '@cat-cafe/shared';
import { getCatModel } from '../../../../../config/cat-models.js';
import {
  isFunctionCallingUnsupportedGeminiModel,
  normalizeGeminiModelName,
} from '../../../../../config/gemini-model-capabilities.js';
import { createModuleLogger } from '../../../../../infrastructure/logger.js';
import { formatCliExitError } from '../../../../../utils/cli-format.js';
import { formatCliNotFoundError, resolveCliCommand } from '../../../../../utils/cli-resolve.js';
import {
  buildChildEnv,
  isCliError,
  isCliTimeout,
  isLivenessWarning,
  spawnCli,
} from '../../../../../utils/cli-spawn.js';
import type { SpawnFn } from '../../../../../utils/cli-types.js';
import type { AgentMessage, AgentService, AgentServiceOptions, MessageMetadata, TokenUsage } from '../../types.js';
import { appendLocalImagePathHints, collectImageAccessDirectories } from '../providers/image-cli-bridge.js';
import { extractImagePaths } from '../providers/image-paths.js';
import { isKnownPostResponseCandidatesCrash, isResultErrorEvent, transformGeminiEvent } from './gemini-event-parser.js';

const log = createModuleLogger('gemini-agent');

type GeminiAdapter = 'gemini-cli' | 'antigravity';

const GEMINI_DISABLED_MCP_SENTINEL = '__cat_cafe_no_mcp__';
const GEMINI_IMAGE_MIME_WHITELIST = new Set(['image/png', 'image/jpeg', 'image/webp']);

interface GeminiStoredThought {
  readonly subject?: string;
  readonly description?: string;
}

interface GeminiStoredMessage {
  readonly type?: string;
  readonly content?: string;
  readonly thoughts?: readonly GeminiStoredThought[];
}

interface GeminiStoredSession {
  readonly sessionId?: string;
  readonly messages?: readonly GeminiStoredMessage[];
}

function normalizeGeminiContent(value: string | undefined): string {
  return (value ?? '').replace(/\s+/g, ' ').trim();
}

function formatGeminiThoughts(thoughts: readonly GeminiStoredThought[]): string {
  return thoughts
    .map((thought) => {
      const subject = thought.subject?.trim();
      const description = thought.description?.trim();
      if (subject && description) return `**${subject}**\n${description}`;
      if (subject) return `**${subject}**`;
      if (description) return description;
      return '';
    })
    .filter((chunk) => chunk.length > 0)
    .join('\n\n---\n\n');
}

function sanitizePathSegment(value: string | undefined): string {
  const trimmed = value?.trim();
  if (!trimmed) return 'default';
  const sanitized = trimmed.replace(/[^a-zA-Z0-9._-]+/g, '-').replace(/^-+|-+$/g, '');
  return sanitized.length > 0 ? sanitized.slice(0, 48) : 'default';
}

function shouldIsolateGeminiHome(callbackEnv: Record<string, string>, model: string | undefined): boolean {
  const hasApiKey = Boolean(callbackEnv.GEMINI_API_KEY || callbackEnv.GOOGLE_API_KEY);
  const hasCustomEndpoint = Boolean(callbackEnv.GOOGLE_VERTEX_BASE_URL || callbackEnv.GOOGLE_GEMINI_BASE_URL);
  return (
    isFunctionCallingUnsupportedGeminiModel(model) ||
    (hasApiKey && (callbackEnv.GOOGLE_GENAI_USE_VERTEXAI === 'true' || hasCustomEndpoint))
  );
}

function normalizeGeminiCallbackEnv(callbackEnv: Record<string, string> | undefined): Record<string, string> | undefined {
  if (!callbackEnv) return undefined;

  const normalizedEnv: Record<string, string> = { ...callbackEnv };
  const legacyBaseUrl = normalizedEnv.GEMINI_BASE_URL?.trim();
  if (legacyBaseUrl) {
    normalizedEnv.GOOGLE_VERTEX_BASE_URL ??= legacyBaseUrl;
    normalizedEnv.GOOGLE_GENAI_USE_VERTEXAI ??= 'true';
  }
  const vertexBaseUrl = normalizedEnv.GOOGLE_VERTEX_BASE_URL?.trim();
  if (vertexBaseUrl && !normalizedEnv.GOOGLE_GENAI_API_VERSION) {
    // Gemini CLI defaults Vertex API calls to v1beta1, but ZenMux-style
    // Vertex gateways expose the GA v1 path.
    normalizedEnv.GOOGLE_GENAI_API_VERSION = 'v1';
  }
  return normalizedEnv;
}

function resolveGeminiHome(
  callbackEnv: Record<string, string>,
  catId: CatId,
  workingDirectory?: string,
  model?: string,
): string | null {
  if (!shouldIsolateGeminiHome(callbackEnv, model)) return null;

  const endpoint = callbackEnv.GOOGLE_VERTEX_BASE_URL ?? callbackEnv.GOOGLE_GEMINI_BASE_URL ?? 'default-endpoint';
  const mode = isFunctionCallingUnsupportedGeminiModel(model) ? 'no-tools' : 'default';
  const scope = `${catId}:${workingDirectory ?? ''}:${endpoint}:${normalizeGeminiModelName(model)}:${mode}`;
  const hash = createHash('sha256').update(scope).digest('hex').slice(0, 12);
  const catSegment = sanitizePathSegment(String(catId));
  const projectSegment = sanitizePathSegment(workingDirectory ? basename(workingDirectory) : undefined);
  const isolatedHome = join(tmpdir(), `cat-cafe-gemini-home-${catSegment}-${projectSegment}-${hash}`);
  mkdirSync(join(isolatedHome, '.gemini', 'tmp'), { recursive: true });
  return isolatedHome;
}

function ensureImageModelGeminiSettings(homeRoot: string, model: string): void {
  const geminiDir = join(homeRoot, '.gemini');
  const settingsPath = join(geminiDir, 'settings.json');
  mkdirSync(geminiDir, { recursive: true });

  let settings: Record<string, unknown> = {};
  if (existsSync(settingsPath)) {
    try {
      const parsed = JSON.parse(readFileSync(settingsPath, 'utf8')) as Record<string, unknown>;
      if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
        settings = parsed;
      }
    } catch {
      // Overwrite malformed settings in isolated HOME with a deterministic profile.
    }
  }

  const tools =
    settings.tools && typeof settings.tools === 'object' && !Array.isArray(settings.tools)
      ? ({ ...(settings.tools as Record<string, unknown>) } as Record<string, unknown>)
      : {};
  tools.core = [];
  const exclude = Array.isArray(tools.exclude) ? [...tools.exclude] : [];
  for (const toolName of ['activate_skill', 'run_shell_command']) {
    if (!exclude.includes(toolName)) exclude.push(toolName);
  }
  tools.exclude = exclude;
  settings.tools = tools;

  const experimental =
    settings.experimental && typeof settings.experimental === 'object' && !Array.isArray(settings.experimental)
      ? ({ ...(settings.experimental as Record<string, unknown>) } as Record<string, unknown>)
      : {};
  experimental.enableAgents = false;
  settings.experimental = experimental;

  // Gemini CLI falls back to `chat-base` alias for unknown chat models, which
  // injects `thinkingConfig: { includeThoughts: true }`.  Image-generation
  // models reject thinking entirely (400).  Defining a custom alias for the
  // model prevents the chat-base fallback and keeps thinkingConfig out of the
  // API request.
  const noThinkingAlias = {
    modelConfig: {
      model,
      generateContentConfig: { temperature: 0.7, topP: 0.95 },
    },
  };
  const modelConfigs =
    settings.modelConfigs && typeof settings.modelConfigs === 'object' && !Array.isArray(settings.modelConfigs)
      ? ({ ...(settings.modelConfigs as Record<string, unknown>) } as Record<string, unknown>)
      : {};
  const customAliases =
    modelConfigs.customAliases && typeof modelConfigs.customAliases === 'object' && !Array.isArray(modelConfigs.customAliases)
      ? ({ ...(modelConfigs.customAliases as Record<string, unknown>) } as Record<string, unknown>)
      : {};
  customAliases[model] = noThinkingAlias;
  // Also register the bare model name (without provider prefix) so resolution
  // matches regardless of how the CLI normalises the `--model` value.
  const bare = normalizeGeminiModelName(model);
  if (bare && bare !== model) {
    customAliases[bare] = { ...noThinkingAlias, modelConfig: { ...noThinkingAlias.modelConfig, model } };
  }
  modelConfigs.customAliases = customAliases;
  settings.modelConfigs = modelConfigs;

  const skills =
    settings.skills && typeof settings.skills === 'object' && !Array.isArray(settings.skills)
      ? ({ ...(settings.skills as Record<string, unknown>) } as Record<string, unknown>)
      : {};
  skills.enabled = false;
  settings.skills = skills;

  const admin =
    settings.admin && typeof settings.admin === 'object' && !Array.isArray(settings.admin)
      ? ({ ...(settings.admin as Record<string, unknown>) } as Record<string, unknown>)
      : {};
  const adminSkills =
    admin.skills && typeof admin.skills === 'object' && !Array.isArray(admin.skills)
      ? ({ ...(admin.skills as Record<string, unknown>) } as Record<string, unknown>)
      : {};
  adminSkills.enabled = false;
  admin.skills = adminSkills;
  settings.admin = admin;

  writeFileSync(settingsPath, `${JSON.stringify(settings, null, 2)}\n`, 'utf8');
}

interface GeminiCliRuntimeConfig {
  readonly env: Record<string, string>;
  readonly extraArgs: readonly string[];
}

function buildGeminiCliRuntimeConfig(
  callbackEnv: Record<string, string> | undefined,
  catId: CatId,
  model: string,
  workingDirectory?: string,
): GeminiCliRuntimeConfig | undefined {
  const normalizedEnv = normalizeGeminiCallbackEnv(callbackEnv);
  if (!normalizedEnv) return undefined;

  const isolatedHome = resolveGeminiHome(normalizedEnv, catId, workingDirectory, model);
  if (!isolatedHome) {
    return { env: normalizedEnv, extraArgs: [] };
  }

  normalizedEnv.HOME = isolatedHome;
  if (process.platform === 'win32') {
    normalizedEnv.USERPROFILE = isolatedHome;
  }
  normalizedEnv.GOOGLE_GENAI_USE_GCA = 'false';
  normalizedEnv.CLOUD_SHELL = 'false';
  normalizedEnv.GEMINI_CLI_USE_COMPUTE_ADC = 'false';

  const extraArgs: string[] = [];
  if (isFunctionCallingUnsupportedGeminiModel(model)) {
    // Vertex image-generation models reject function calling, so Gemini CLI
    // must run as a plain model client instead of the default agent profile.
    ensureImageModelGeminiSettings(isolatedHome, model);
    extraArgs.push('--extensions', 'none', '--allowed-mcp-server-names', GEMINI_DISABLED_MCP_SENTINEL);
  }

  return { env: normalizedEnv, extraArgs };
}

function inferGeminiInlineMimeType(imagePath: string): string | null {
  switch (extname(imagePath).toLowerCase()) {
    case '.png':
      return 'image/png';
    case '.jpg':
    case '.jpeg':
      return 'image/jpeg';
    case '.webp':
      return 'image/webp';
    default:
      return null;
  }
}

function buildGeminiImagePromptParts(prompt: string, imagePaths: readonly string[]): Array<Record<string, unknown>> {
  const parts: Array<Record<string, unknown>> = [];
  const trimmedPrompt = prompt.trim();
  if (trimmedPrompt.length > 0) {
    parts.push({ text: trimmedPrompt });
  }

  for (const imagePath of imagePaths) {
    const mimeType = inferGeminiInlineMimeType(imagePath);
    if (!mimeType) continue;
    try {
      const data = readFileSync(imagePath).toString('base64');
      parts.push({ inlineData: { mimeType, data } });
    } catch {
      // Fall back to prompt-only generation if a local image cannot be read.
    }
  }

  return parts.length > 0 ? parts : [{ text: prompt }];
}

function extractGeminiUsageMetadata(raw: unknown): TokenUsage | undefined {
  if (typeof raw !== 'object' || raw === null) return undefined;
  const usage = raw as Record<string, unknown>;
  const tokenUsage: TokenUsage = {};
  if (typeof usage.totalTokenCount === 'number') tokenUsage.totalTokens = usage.totalTokenCount;
  if (typeof usage.promptTokenCount === 'number') tokenUsage.inputTokens = usage.promptTokenCount;
  if (typeof usage.candidatesTokenCount === 'number') tokenUsage.outputTokens = usage.candidatesTokenCount;
  if (typeof usage.cachedContentTokenCount === 'number') tokenUsage.cacheReadTokens = usage.cachedContentTokenCount;
  return Object.keys(tokenUsage).length > 0 ? tokenUsage : undefined;
}

function buildGeminiImageRichBlock(catId: CatId, imageItems: Array<{ url: string; alt: string }>): AgentMessage {
  return {
    type: 'system_info',
    catId,
    content: JSON.stringify({
      type: 'rich_block',
      block: {
        id: `gemini-img-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
        kind: 'media_gallery',
        v: 1,
        title: 'gemini:image-generation',
        items: imageItems,
      },
    }),
    timestamp: Date.now(),
  };
}

function readGeminiThinkingFromLocalSession(
  sessionId: string | undefined,
  assistantText: string,
  workingDirectory?: string,
  homeRoot?: string,
): string | null {
  if (!sessionId) return null;

  const geminiTmpRoot = join(homeRoot ?? homedir(), '.gemini', 'tmp');
  if (!existsSync(geminiTmpRoot)) return null;

  const preferredProjectDir = workingDirectory ? basename(workingDirectory) : null;
  const projectDirs = readdirSync(geminiTmpRoot, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => entry.name)
    .sort((a, b) => {
      if (preferredProjectDir && a === preferredProjectDir) return -1;
      if (preferredProjectDir && b === preferredProjectDir) return 1;
      return 0;
    });

  const normalizedAssistantText = normalizeGeminiContent(assistantText);

  for (const projectDir of projectDirs) {
    const chatsDir = join(geminiTmpRoot, projectDir, 'chats');
    if (!existsSync(chatsDir)) continue;

    const sessionFiles = readdirSync(chatsDir)
      .filter((name) => name.startsWith('session-') && name.endsWith('.json'))
      .map((name) => ({
        path: join(chatsDir, name),
        mtimeMs: statSync(join(chatsDir, name)).mtimeMs,
      }))
      .sort((a, b) => b.mtimeMs - a.mtimeMs);

    for (const file of sessionFiles) {
      try {
        const parsed = JSON.parse(readFileSync(file.path, 'utf8')) as GeminiStoredSession;
        if (parsed.sessionId !== sessionId || !Array.isArray(parsed.messages)) continue;

        const candidates = parsed.messages.filter(
          (message): message is GeminiStoredMessage =>
            message?.type === 'gemini' &&
            Array.isArray(message.thoughts) &&
            message.thoughts.length > 0 &&
            typeof message.content === 'string',
        );
        if (candidates.length === 0) return null;

        const exact =
          normalizedAssistantText.length > 0
            ? [...candidates]
                .reverse()
                .find((message) => normalizeGeminiContent(message.content) === normalizedAssistantText)
            : candidates[candidates.length - 1];
        const selected = exact ?? null;
        return selected ? formatGeminiThoughts(selected.thoughts ?? []) || null : null;
      } catch {
        // Best effort: skip malformed/partial session files while Gemini is still writing them.
      }
    }
  }

  return null;
}
/**
 * Options for constructing GeminiAgentService (dependency injection)
 * F32-b: catId and model are constructor parameters
 */
interface GeminiAgentServiceOptions {
  /** F32-b: catId for this instance (default: 'gemini') */
  catId?: CatId;
  /** F32-b: model override (default: resolved via getCatModel) */
  model?: string;
  /** Inject spawn for gemini-cli adapter (via spawnCli) */
  spawnFn?: SpawnFn;
  /** Inject spawn for antigravity adapter (direct child_process.spawn) */
  antigravitySpawnFn?: typeof nodeSpawn;
  /** Override adapter selection (default: GEMINI_ADAPTER env or 'gemini-cli') */
  adapter?: GeminiAdapter;
  /** Inject fetch for direct Vertex image-generation path */
  fetchFn?: typeof fetch;
}

/**
 * Service for invoking Gemini via CLI subprocess (dual adapter).
 * Uses Google AI Pro/Ultra subscription instead of API key.
 */
export class GeminiAgentService implements AgentService {
  readonly catId: CatId;
  private readonly spawnFn: SpawnFn | undefined;
  private readonly model: string;
  private readonly antigravitySpawnFn: typeof nodeSpawn;
  private readonly adapter: GeminiAdapter;
  private readonly fetchFn: typeof fetch;
  constructor(options?: GeminiAgentServiceOptions) {
    this.catId = options?.catId ?? createCatId('gemini');
    this.model = options?.model ?? getCatModel(this.catId as string);
    this.spawnFn = options?.spawnFn;
    this.antigravitySpawnFn = options?.antigravitySpawnFn ?? nodeSpawn;
    this.adapter = options?.adapter ?? (process.env.GEMINI_ADAPTER as GeminiAdapter | undefined) ?? 'gemini-cli';
    this.fetchFn = options?.fetchFn ?? fetch;
  }

  async *invoke(prompt: string, options?: AgentServiceOptions): AsyncIterable<AgentMessage> {
    if (this.adapter === 'antigravity') {
      yield* this.invokeAntigravity(prompt, options);
    } else {
      yield* this.invokeGeminiCLI(prompt, options);
    }
  }

  private async *invokeGeminiCLI(prompt: string, options?: AgentServiceOptions): AsyncIterable<AgentMessage> {
    const effectiveModel = options?.callbackEnv?.CAT_CAFE_GEMINI_MODEL_OVERRIDE ?? this.model;
    const metadata: MessageMetadata = { provider: 'google', model: effectiveModel };
    const isImageGenerationModel = isFunctionCallingUnsupportedGeminiModel(effectiveModel);

    // Gemini CLI has no system prompt flag; prepend identity to prompt text
    let effectivePrompt = options?.systemPrompt ? `${options.systemPrompt}\n\n${prompt}` : prompt;

    const imagePaths = extractImagePaths(options?.contentBlocks, options?.uploadDir);
    const imageAccessDirs = collectImageAccessDirectories(imagePaths);
    // Gemini CLI -i is prompt-interactive (conflicts with -p), so we pass path hints
    // and include image directories for tool access.
    effectivePrompt = appendLocalImagePathHints(effectivePrompt, imagePaths);

    if (isImageGenerationModel) {
      yield* this.invokeGeminiImageApi(effectivePrompt, effectiveModel, metadata, imagePaths, options);
      return;
    }

    // Gemini CLI supports UUID session resume in headless mode:
    //   gemini --resume <sessionId> -p "<prompt>" -o stream-json
    // Prefer resume when sessionId is available so Gemini follows the same
    // session semantics as Claude/Codex (session-chain + self-heal).
    const cliRuntime = buildGeminiCliRuntimeConfig(options?.callbackEnv, this.catId, effectiveModel, options?.workingDirectory);
    const modelArgs = ['--model', effectiveModel, ...(cliRuntime?.extraArgs ?? [])];
    // Vertex image-generation models break when old tool-call history is
    // replayed, and Gemini CLI persists tool responses verbatim (including
    // functionResponse.id). Force fresh sessions for these models.
    const args: string[] = options?.sessionId && !isImageGenerationModel
      ? ['--resume', options?.sessionId!, ...modelArgs, '-p', effectivePrompt, '-o', 'stream-json', '-y']
      : [...modelArgs, '-p', effectivePrompt, '-o', 'stream-json', '-y'];
    for (const dir of imageAccessDirs) {
      args.push('--include-directories', dir);
    }

    try {
      const geminiCommand = resolveCliCommand('gemini');
      if (!geminiCommand) {
        yield {
          type: 'error' as const,
          catId: this.catId,
          error: formatCliNotFoundError('gemini'),
          metadata,
          timestamp: Date.now(),
        };
        yield { type: 'done' as const, catId: this.catId, metadata, timestamp: Date.now() };
        return;
      }

      let sawResultError = false;
      let sawAssistantText = false;
      let suppressCliExitError = false;
      let fullAssistantText = '';
      const cliOpts = {
        command: geminiCommand,
        args,
        ...(options?.workingDirectory ? { cwd: options.workingDirectory } : {}),
        ...(cliRuntime ? { env: cliRuntime.env } : {}),
        ...(options?.signal ? { signal: options.signal } : {}),
        ...(options?.invocationId ? { invocationId: options.invocationId } : {}),
        ...(options?.cliSessionId ? { cliSessionId: options.cliSessionId } : {}),
        ...(options?.livenessProbe ? { livenessProbe: options.livenessProbe } : {}),
        ...(options?.parentSpan ? { parentSpan: options.parentSpan } : {}),
      };
      const events = options?.spawnCliOverride
        ? options.spawnCliOverride(cliOpts)
        : spawnCli(cliOpts, this.spawnFn ? { spawnFn: this.spawnFn } : undefined);

      for await (const event of events) {
        if (isCliTimeout(event)) {
          yield {
            type: 'system_info' as const,
            catId: this.catId,
            content: JSON.stringify({
              type: 'timeout_diagnostics',
              silenceDurationMs: event.silenceDurationMs,
              processAlive: event.processAlive,
              lastEventType: event.lastEventType,
              firstEventAt: event.firstEventAt,
              lastEventAt: event.lastEventAt,
              cliSessionId: event.cliSessionId,
              invocationId: event.invocationId,
              rawArchivePath: event.rawArchivePath,
            }),
            timestamp: Date.now(),
          };
          yield {
            type: 'error',
            catId: this.catId,
            error: `暹罗猫 CLI 响应超时 (${Math.round(event.timeoutMs / 1000)}s${event.firstEventAt == null ? ', 未收到首帧' : ''})`,
            metadata,
            timestamp: Date.now(),
          };
          continue;
        }
        // F118 Phase C: Forward liveness warnings to frontend with catId
        if (isLivenessWarning(event)) {
          const warningEvent = event as { level?: string; silenceDurationMs?: number };
          log.warn(
            {
              catId: this.catId,
              invocationId: options?.invocationId,
              level: warningEvent.level,
              silenceMs: warningEvent.silenceDurationMs,
            },
            '[GeminiAgent] liveness warning — CLI may be stuck',
          );
          yield {
            type: 'system_info' as const,
            catId: this.catId,
            content: JSON.stringify({ type: 'liveness_warning', ...event }),
            timestamp: Date.now(),
          };
          continue;
        }
        if (isCliError(event)) {
          if (sawResultError || suppressCliExitError) continue;
          yield {
            type: 'error',
            catId: this.catId,
            error: formatCliExitError('Gemini CLI', event),
            metadata,
            timestamp: Date.now(),
          };
          continue;
        }

        // F8: Capture usage from result/success events before transform drops them
        if (typeof event === 'object' && event !== null) {
          const raw = event as Record<string, unknown>;
          if (raw.type === 'result' && raw.status === 'success') {
            const stats = raw.stats as Record<string, unknown> | undefined;
            if (stats) {
              const usage: TokenUsage = {};
              if (typeof stats.total_tokens === 'number') usage.totalTokens = stats.total_tokens;
              if (typeof stats.input_tokens === 'number') usage.inputTokens = stats.input_tokens;
              if (typeof stats.output_tokens === 'number') usage.outputTokens = stats.output_tokens;
              if (typeof stats.cached_input_tokens === 'number') usage.cacheReadTokens = stats.cached_input_tokens;
              const contextWindow =
                (typeof stats.context_window === 'number' ? stats.context_window : undefined) ??
                (typeof stats.contextWindow === 'number' ? stats.contextWindow : undefined);
              if (contextWindow != null) usage.contextWindowSize = contextWindow;
              metadata.usage = usage;
            }
          }
        }

        if (sawAssistantText && isKnownPostResponseCandidatesCrash(event)) {
          suppressCliExitError = true;
          continue;
        }

        const fromResultError = isResultErrorEvent(event);
        const result = transformGeminiEvent(event, this.catId);
        if (result !== null) {
          if (result.type === 'session_init' && result.sessionId) {
            metadata.sessionId = result.sessionId;
          }
          if (result.type === 'text') {
            // Separate consecutive assistant text turns with paragraph break.
            // Each Gemini message/assistant is a complete turn (unlike Claude's
            // incremental deltas), so direct concatenation loses inter-turn spacing.
            if (sawAssistantText && result.content) {
              fullAssistantText += `\n\n${result.content}`;
              yield { ...result, content: `\n\n${result.content}`, metadata };
            } else {
              fullAssistantText += result.content ?? '';
              yield { ...result, metadata };
            }
            sawAssistantText = true;
          } else {
            if (fromResultError && result.type === 'error') {
              sawResultError = true;
            }
            yield { ...result, metadata };
          }
        }
      }

      const thinking = readGeminiThinkingFromLocalSession(
        metadata.sessionId,
        fullAssistantText,
        options?.workingDirectory,
        cliRuntime?.env.HOME,
      );
      if (thinking) {
        yield {
          type: 'system_info',
          catId: this.catId,
          content: JSON.stringify({ type: 'thinking', catId: this.catId, text: thinking }),
          metadata,
          timestamp: Date.now(),
        };
      }

      yield { type: 'done', catId: this.catId, metadata, timestamp: Date.now() };
    } catch (err) {
      yield {
        type: 'error',
        catId: this.catId,
        error: err instanceof Error ? err.message : String(err),
        metadata,
        timestamp: Date.now(),
      };
      // Guarantee done after error so invoke-single-cat can set isFinal correctly
      yield { type: 'done', catId: this.catId, metadata, timestamp: Date.now() };
    }
  }

  private async *invokeGeminiImageApi(
    prompt: string,
    effectiveModel: string,
    metadata: MessageMetadata,
    imagePaths: readonly string[],
    options?: AgentServiceOptions,
  ): AsyncIterable<AgentMessage> {
    const normalizedEnv = normalizeGeminiCallbackEnv(options?.callbackEnv);
    const apiKey = normalizedEnv?.GOOGLE_API_KEY?.trim() || normalizedEnv?.GEMINI_API_KEY?.trim();
    const vertexBaseUrl = normalizedEnv?.GOOGLE_VERTEX_BASE_URL?.trim();
    const apiVersion = normalizedEnv?.GOOGLE_GENAI_API_VERSION?.trim() || 'v1';

    if (!apiKey || !vertexBaseUrl) {
      yield {
        type: 'error',
        catId: this.catId,
        error: 'Gemini image-generation models require GOOGLE_API_KEY/GEMINI_API_KEY and GOOGLE_VERTEX_BASE_URL.',
        metadata,
        timestamp: Date.now(),
      };
      yield { type: 'done', catId: this.catId, metadata, timestamp: Date.now() };
      return;
    }

    const sessionId = `vertex-image-${randomUUID()}`;
    metadata.sessionId = sessionId;
    yield {
      type: 'session_init',
      catId: this.catId,
      sessionId,
      ephemeralSession: true,
      metadata,
      timestamp: Date.now(),
    };

    const modelName = normalizeGeminiModelName(effectiveModel);
    const requestUrl = `${vertexBaseUrl.replace(/\/+$/, '')}/${apiVersion}/publishers/google/models/${modelName}:generateContent`;
    const requestBody = {
      contents: [{ role: 'user', parts: buildGeminiImagePromptParts(prompt, imagePaths) }],
      generationConfig: {
        responseModalities: ['TEXT', 'IMAGE'],
        temperature: 0.7,
        topP: 0.95,
      },
    };

    try {
      const response = await this.fetchFn(requestUrl, {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          'x-goog-api-key': apiKey,
        },
        body: JSON.stringify(requestBody),
        signal: options?.signal,
      });

      if (!response.ok) {
        const body = await response.text().catch(() => '');
        throw new Error(`Vertex image API error ${response.status}: ${body.trim() || response.statusText}`);
      }

      const payload = (await response.json()) as Record<string, unknown>;
      metadata.usage = extractGeminiUsageMetadata(payload.usageMetadata);

      const candidates = Array.isArray(payload.candidates) ? (payload.candidates as Array<Record<string, unknown>>) : [];
      const parts = candidates.flatMap((candidate) => {
        const content = candidate.content;
        if (typeof content !== 'object' || content === null) return [];
        const candidateParts = (content as Record<string, unknown>).parts;
        return Array.isArray(candidateParts) ? (candidateParts as Array<Record<string, unknown>>) : [];
      });

      const textChunks = parts
        .map((part) => (typeof part.text === 'string' ? part.text.trim() : ''))
        .filter((chunk) => chunk.length > 0);
      if (textChunks.length > 0) {
        yield {
          type: 'text',
          catId: this.catId,
          content: textChunks.join('\n\n'),
          metadata,
          timestamp: Date.now(),
        };
      }

      const imageItems = parts
        .map((part) => {
          const inlineData = part.inlineData;
          if (typeof inlineData !== 'object' || inlineData === null) return null;
          const typedInlineData = inlineData as Record<string, unknown>;
          const mimeType = typeof typedInlineData.mimeType === 'string' ? typedInlineData.mimeType : '';
          const data = typeof typedInlineData.data === 'string' ? typedInlineData.data : '';
          if (!GEMINI_IMAGE_MIME_WHITELIST.has(mimeType) || data.length === 0) return null;
          return { url: `data:${mimeType};base64,${data}`, alt: 'Gemini generated image' };
        })
        .filter((item): item is { url: string; alt: string } => item !== null);
      if (imageItems.length > 0) {
        yield { ...buildGeminiImageRichBlock(this.catId, imageItems), metadata };
      }

      if (textChunks.length === 0 && imageItems.length === 0) {
        yield {
          type: 'error',
          catId: this.catId,
          error: 'Gemini image API returned no text or image parts.',
          metadata,
          timestamp: Date.now(),
        };
      }
    } catch (err) {
      yield {
        type: 'error',
        catId: this.catId,
        error: err instanceof Error ? err.message : String(err),
        metadata,
        timestamp: Date.now(),
      };
    }

    yield { type: 'done', catId: this.catId, metadata, timestamp: Date.now() };
  }

  private async *invokeAntigravity(prompt: string, options?: AgentServiceOptions): AsyncIterable<AgentMessage> {
    const agMetadata: MessageMetadata = { provider: 'google', model: `${this.model} (antigravity)` };

    if (!options?.callbackEnv) {
      yield {
        type: 'error',
        catId: this.catId,
        error: 'antigravity adapter requires callbackEnv for MCP callback',
        metadata: agMetadata,
        timestamp: Date.now(),
      };
      yield { type: 'done', catId: this.catId, metadata: agMetadata, timestamp: Date.now() };
      return;
    }

    const sessionId = `antigravity-${randomUUID()}`;
    agMetadata.sessionId = sessionId;
    yield {
      type: 'session_init',
      catId: this.catId,
      sessionId,
      metadata: agMetadata,
      timestamp: Date.now(),
    };

    let spawnError: Error | null = null;

    try {
      // Clone all env, strip bloated vars (LS_COLORS etc.) to avoid E2BIG,
      // then merge callbackEnv overrides. Preserves API keys etc. from parent env.
      const childEnv = buildChildEnv(options.callbackEnv);

      const child = this.antigravitySpawnFn('antigravity', ['chat', '--mode', 'agent', prompt], {
        detached: true,
        stdio: 'ignore',
        env: childEnv as Record<string, string>,
      });
      // Capture async spawn errors (ENOENT etc.) that fire on next tick.
      child.on('error', (err: Error) => {
        spawnError = err;
      });

      // Wire AbortSignal to kill the detached process group
      const pid = child.pid;
      if (pid && options?.signal) {
        options.signal.addEventListener(
          'abort',
          () => {
            try {
              process.kill(-pid, 'SIGTERM');
              log.debug({ pid }, `[gemini] Antigravity process group killed via signal`);
            } catch {
              /* already exited */
            }
          },
          { once: true },
        );
      }

      child.unref();
    } catch (err) {
      yield {
        type: 'error',
        catId: this.catId,
        error: `Failed to launch Antigravity: ${err instanceof Error ? err.message : String(err)}`,
        metadata: agMetadata,
        timestamp: Date.now(),
      };
      yield { type: 'done', catId: this.catId, metadata: agMetadata, timestamp: Date.now() };
      return;
    }

    // Wait one tick — most spawn errors (ENOENT, EACCES) fire here.
    await new Promise((resolve) => process.nextTick(resolve));

    if (spawnError) {
      yield {
        type: 'error',
        catId: this.catId,
        error: `Failed to launch Antigravity: ${(spawnError as Error).message}`,
        metadata: agMetadata,
        timestamp: Date.now(),
      };
      yield { type: 'done', catId: this.catId, metadata: agMetadata, timestamp: Date.now() };
      return;
    }

    yield {
      type: 'text',
      catId: this.catId,
      content: '暹罗猫已在 Antigravity 中开始工作，结果将通过 MCP 回传到对话中。',
      metadata: agMetadata,
      timestamp: Date.now(),
    };

    yield { type: 'done', catId: this.catId, metadata: agMetadata, timestamp: Date.now() };
  }
}
