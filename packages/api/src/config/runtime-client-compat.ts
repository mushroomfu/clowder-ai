import type { CatConfig, ClientId } from '@cat-cafe/shared';
import type { RuntimeProviderProfile } from './account-resolver.js';

type CompatCatConfig = Pick<CatConfig, 'clientId' | 'defaultModel'>;
type CompatRuntimeProfile = Pick<RuntimeProviderProfile, 'authType' | 'baseUrl' | 'models'>;

/**
 * Compatibility shim for legacy runtime variants that were created with an
 * Anthropic client purely to bypass the older "google builtin only" guard.
 *
 * Signature of the bad data:
 * - clientId === anthropic
 * - model looks like google/*
 * - bound api_key account looks like a Vertex/Gemini gateway
 */
export function resolveCompatibleClientId(
  catConfig: CompatCatConfig | null | undefined,
  runtimeProfile?: CompatRuntimeProfile | null,
): ClientId | undefined {
  const configuredClient = catConfig?.clientId;
  if (!configuredClient) return undefined;
  if (configuredClient !== 'anthropic') return configuredClient;

  const model = catConfig?.defaultModel?.trim() ?? '';
  if (!model.startsWith('google/')) return configuredClient;

  if (runtimeProfile?.authType !== 'api_key') return configuredClient;

  const hasGoogleModel = runtimeProfile.models?.some((entry) => entry.startsWith('google/')) ?? false;
  const baseUrl = runtimeProfile.baseUrl?.toLowerCase() ?? '';
  const looksLikeVertexGateway = baseUrl.includes('vertex-ai') || baseUrl.includes('/models/');

  if (hasGoogleModel || looksLikeVertexGateway) {
    return 'google';
  }

  return configuredClient;
}
