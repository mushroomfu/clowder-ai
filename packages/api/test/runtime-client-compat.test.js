import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

describe('runtime-client-compat', () => {
  it('reroutes legacy anthropic google-model variants to google for Vertex api_key accounts', async () => {
    const { resolveCompatibleClientId } = await import(`../dist/config/runtime-client-compat.js?t=${Date.now()}`);

    const clientId = resolveCompatibleClientId(
      {
        clientId: 'anthropic',
        defaultModel: 'google/gemini-3-pro-image-preview',
      },
      {
        authType: 'api_key',
        baseUrl: 'https://zenmux.ai/api/vertex-ai',
        models: ['google/gemini-3-pro-image-preview'],
      },
    );

    assert.equal(clientId, 'google');
  });

  it('keeps anthropic variants on anthropic when the bound account does not look like a google gateway', async () => {
    const { resolveCompatibleClientId } = await import(`../dist/config/runtime-client-compat.js?t=${Date.now()}-2`);

    const clientId = resolveCompatibleClientId(
      {
        clientId: 'anthropic',
        defaultModel: 'google/gemini-3-pro-image-preview',
      },
      {
        authType: 'api_key',
        baseUrl: 'https://api.anthropic.com',
        models: ['claude-opus-4-6'],
      },
    );

    assert.equal(clientId, 'anthropic');
  });
});
