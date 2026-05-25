import { createContext, useCallback, useContext, useMemo, useState } from "react";

type ToastFn = (message: string, isError?: boolean) => void;

const ToastContext = createContext<ToastFn>(() => {});

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toast, setToast] = useState<{ message: string; isError: boolean } | null>(null);

  const showToast = useCallback((message: string, isError = false) => {
    setToast({ message, isError });
    window.setTimeout(() => setToast(null), 2800);
  }, []);

  const value = useMemo(() => showToast, [showToast]);

  return (
    <ToastContext.Provider value={value}>
      {children}
      <div className={`toast ${toast ? "show" : ""} ${toast?.isError ? "error" : ""}`}>
        {toast?.message}
      </div>
    </ToastContext.Provider>
  );
}

export function useToast() {
  return useContext(ToastContext);
}
