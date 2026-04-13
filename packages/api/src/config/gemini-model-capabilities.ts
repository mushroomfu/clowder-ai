export function normalizeGeminiModelName(model: string | undefined): string {
  const trimmed = model?.trim();
  if (!trimmed) return '';
  const slashIndex = trimmed.lastIndexOf('/');
  return slashIndex >= 0 ? trimmed.slice(slashIndex + 1) : trimmed;
}

export function isFunctionCallingUnsupportedGeminiModel(model: string | undefined): boolean {
  const normalized = normalizeGeminiModelName(model).toLowerCase();
  if (!normalized) return false;

  return (
    normalized === 'gemini-3-pro-image-preview' ||
    normalized.endsWith('-pro-image-preview') ||
    normalized.endsWith('-flash-image') ||
    normalized.endsWith('-flash-image-preview')
  );
}
