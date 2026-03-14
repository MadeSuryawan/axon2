import { useState, useEffect, useRef } from 'react';
import { searchApi } from '@/api/client';
import type { SearchResult } from '@/types';

export function useSearch(query: string, enabled: boolean) {
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const timerRef = useRef<ReturnType<typeof setTimeout>>();
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    if (!enabled || !query.trim()) {
      setResults([]);
      return;
    }

    clearTimeout(timerRef.current);
    timerRef.current = setTimeout(async () => {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;
      setLoading(true);
      try {
        const data = await searchApi.search(query, 10, { signal: controller.signal });
        setResults(data.results);
      } catch (err) {
        if (err instanceof DOMException && err.name === 'AbortError') return;
        setResults([]);
      } finally {
        setLoading(false);
      }
    }, 200);

    return () => {
      clearTimeout(timerRef.current);
      abortRef.current?.abort();
    };
  }, [query, enabled]);

  return { results, loading };
}
