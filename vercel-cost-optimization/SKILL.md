---
name: vercel-cost-optimization
description: Use when deploying Next.js apps to Vercel and costs are high, or when setting up a new Vercel project. Covers ISR-breaking patterns, function constraints, caching, Fluid Compute, build optimization. Triggers: "Vercel bill", "Vercel cost", "ISR broken", "dynamic rendering", "cache-control private", "x-vercel-cache MISS", "function invocations", "Fluid Compute", "GB-hours", "s-maxage", "stale-while-revalidate", "maxDuration", "build minutes"
---

# Vercel Cost Optimization

## When to Use

- Vercel bill is higher than expected
- All pages show `cache-control: private, no-cache, no-store`
- `x-vercel-cache: MISS` on every request
- Setting up a new Next.js + Vercel project
- Reviewing deployment config before going live
- Investigating why ISR/SSG isn't working

## Quick Diagnosis

```bash
# Check if ISR is working (run twice, second should be HIT)
curl -sI https://your-site.com/ | grep -i 'cache-control\|x-vercel-cache'

# Healthy:
#   cache-control: s-maxage=300, stale-while-revalidate=60
#   x-vercel-cache: HIT

# Broken:
#   cache-control: private, no-cache, no-store, max-age=0, must-revalidate
#   x-vercel-cache: MISS
```

If every page returns `private, no-cache`, something is forcing dynamic rendering.

## ISR-Breaking Patterns (Most Expensive)

### Pattern 1: `cookies()` or `headers()` in Shared Code

Calling `cookies()` or `headers()` from `next/headers` **anywhere in the rendering tree** forces the entire page to be dynamic. This includes root layouts, shared components, and i18n config.

```typescript
// BAD: Forces EVERY page dynamic (root layout runs for all pages)
// src/app/layout.tsx
export default async function RootLayout({ children }) {
  const locale = await getLocale(); // internally calls cookies()
  return <html lang={locale}>{children}</html>;
}

// BAD: i18n config that reads cookies
// src/i18n/request.ts
export default getRequestConfig(async () => {
  const cookieStore = await cookies();  // FORCES ALL PAGES DYNAMIC
  const locale = cookieStore.get("NEXT_LOCALE")?.value || "en";
  return { locale, messages: ... };
});
```

**Fix**: Use the `[locale]` route segment with `next-intl/middleware`'s `createMiddleware`, which passes locale via request context (ISR-compatible) instead of reading cookies at render time.

```typescript
// GOOD: Middleware handles locale detection (ISR-compatible)
// src/middleware.ts
import createMiddleware from "next-intl/middleware";
import { routing } from "@/i18n/routing";
export default createMiddleware(routing);

// GOOD: request.ts uses requestLocale (set by middleware)
// src/i18n/request.ts
export default getRequestConfig(async ({ requestLocale }) => {
  const locale = (await requestLocale) || "en";
  return { locale, messages: ... };
});

// GOOD: Layout uses setRequestLocale for static rendering
// src/app/[locale]/layout.tsx
export function generateStaticParams() {
  return routing.locales.map((locale) => ({ locale }));
}
export default async function Layout({ params }) {
  const { locale } = await params;
  setRequestLocale(locale);  // Enables ISR
  // ...
}
```

**Important**: `createMiddleware` requires pages under a `[locale]` folder. Without it, the middleware rewrites to paths that don't exist → 404 on all pages.

### Pattern 2: Custom Middleware with `NextResponse.rewrite()`

```typescript
// BAD: Cookie/header reads + rewrite breaks ISR
export async function middleware(request) {
  const locale = request.cookies.get("NEXT_LOCALE")?.value;
  const url = new URL(request.url);
  url.pathname = `/${locale}${url.pathname}`;
  return NextResponse.rewrite(url);  // Forces dynamic
}

// GOOD: Use library middleware or simple NextResponse.next()
export async function middleware(request) {
  // Only do redirects, no rewrites
  if (request.nextUrl.pathname.startsWith("/old-path")) {
    return NextResponse.redirect(new URL("/new-path", request.url), 308);
  }
  return NextResponse.next();
}
```

### Pattern 3: Missing `setRequestLocale()` in Layouts

Even with `[locale]` routing, forgetting `setRequestLocale()` keeps pages dynamic:

```typescript
// BAD: No setRequestLocale
export default async function Layout({ params }) {
  const { locale } = await params;
  const messages = await getMessages();  // Dynamic without setRequestLocale
  return <>{children}</>;
}

// GOOD: Enable static rendering
export default async function Layout({ params }) {
  const { locale } = await params;
  setRequestLocale(locale);  // Tell Next.js this can be static
  const messages = await getMessages();
  return <>{children}</>;
}
```

## Function Constraints (Free Insurance)

### vercel.json Configuration

Always set memory and duration caps to prevent runaway costs:

```json
{
  "functions": {
    "src/app/api/heavy-route/route.ts": {
      "memory": 512,
      "maxDuration": 60
    },
    "src/app/api/light-route/route.ts": {
      "memory": 128,
      "maxDuration": 10
    }
  }
}
```

### Route-Level maxDuration Exports

Belt-and-suspenders approach — add to every API route file:

```typescript
// Light routes (DB reads, simple logic)
export const maxDuration = 10;

// Medium routes (multiple DB queries, external API calls)
export const maxDuration = 15;

// Heavy routes (batch processing, cron jobs)
export const maxDuration = 60;
```

### Memory Guidelines

| Route Type | Memory | Examples |
|---|---|---|
| Stubs / simple JSON | 128 MB | Health check, feature flags, disabled endpoints |
| DB reads / external API | 256 MB | List pages, detail pages, search |
| Heavy processing | 512 MB | Cron jobs, batch operations, email sending |
| Never use | 1024 MB | Default if unset — wasteful for most routes |

## Cache-Control Headers on API Routes

```typescript
// Cacheable GET routes (read-only data)
return NextResponse.json(data, {
  headers: {
    "Cache-Control": "public, s-maxage=3600, stale-while-revalidate=600",
  },
});

// Static/rarely-changing data (templates, configs)
return NextResponse.json(data, {
  headers: {
    "Cache-Control": "public, s-maxage=86400, stale-while-revalidate=3600",
  },
});

// Never cache: auth routes, mutations, user-specific data
// (default behavior, no header needed)
```

## Fluid Compute

**Disable unless you specifically need it.** Fluid Compute adds two line items:
- Fluid Active CPU
- Fluid Provisioned Memory

For typical Next.js sites with ISR pages and short API calls, Fluid Compute adds $50-80/month with no benefit. It's designed for sustained workloads (streaming, long-running connections).

**Where to disable**: Vercel Dashboard → Project Settings → Functions → Fluid Compute toggle.

## Build Optimization

### Skip Non-Source Builds

```json
{
  "git": {
    "ignoredBuildStep": "git diff --quiet HEAD^ -- src/ prisma/ package.json vercel.json next.config.ts"
  }
}
```

This skips builds when only docs, scripts, or config files change — saves build minutes.

### Skip Static Generation During Build

```json
{
  "env": {
    "SKIP_BUILD_STATIC_GENERATION": "true"
  }
}
```

Use `fallback: 'blocking'` or ISR to generate pages on-demand instead of at build time.

## Cost Impact Reference

| Issue | Monthly Cost Impact | Fix Difficulty |
|---|---|---|
| `cookies()`/`headers()` in root layout | +$80-100 | Medium (requires [locale] refactor) |
| Fluid Compute enabled unnecessarily | +$50-80 | Easy (dashboard toggle) |
| No `ignoredBuildStep` | +$15-30 | Easy (vercel.json) |
| No function memory/duration caps | +$10-20 | Easy (vercel.json + exports) |
| No Cache-Control on API routes | +$5-15 | Easy (response headers) |
| Default 1024MB function memory | +$5-10 | Easy (vercel.json) |

## Anti-Patterns

| Anti-Pattern | Why It's Bad | Fix |
|---|---|---|
| Reading `cookies()` in root layout | All pages become dynamic | Use `[locale]` segment + `createMiddleware` |
| `NextResponse.rewrite()` in middleware | Breaks ISR for rewritten paths | Use `NextResponse.next()` or library middleware |
| Using `createMiddleware` without `[locale]` folder | 404 on all pages | Add `[locale]` route segment first |
| Removing `cookies()` without alternative locale detection | Breaks language switching | Need full i18n architecture change, not just deletion |
| No `maxDuration` on API routes | Stuck queries run for 60s at 1024MB | Add `export const maxDuration` to every route |
| Fluid Compute for simple sites | $50-80/mo for no benefit | Disable in dashboard |

## Verification Checklist

After deploying optimizations:

```bash
# 1. Check ISR is working (hit twice)
curl -sI https://site.com/ | grep -i cache-control
curl -sI https://site.com/ | grep -i x-vercel-cache
# Second request should show HIT

# 2. Check API caching
curl -sI https://site.com/api/your-route | grep -i cache-control
# Should show s-maxage

# 3. Check no 404s/500s
for path in / /papers /topics /authors; do
  echo "$path: $(curl -s -o /dev/null -w '%{http_code}' https://site.com$path)"
done

# 4. Check language switching still works (if applicable)
curl -sI --cookie "NEXT_LOCALE=zh" https://site.com/ | grep -i 'content-language\|set-cookie'
```

## See Also

- [fail-fast-ml-engineering](../fail-fast-ml-engineering/) — General engineering discipline
- [agents-md-writing](../agents-md-writing/) — Documenting deployment patterns in CLAUDE.md
