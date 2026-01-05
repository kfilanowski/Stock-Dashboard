interface LoadingValueProps {
  loading: boolean;
  children: React.ReactNode;
  className?: string;
  size?: 'sm' | 'md' | 'lg';
}

export function LoadingValue({ loading, children, className = '', size = 'md' }: LoadingValueProps) {
  if (loading) {
    const sizeClasses = {
      sm: 'w-12 h-3',
      md: 'w-16 h-4',
      lg: 'w-24 h-6'
    };
    
    return (
      <div className={`${sizeClasses[size]} bg-white/10 rounded animate-pulse ${className}`} />
    );
  }
  
  return <>{children}</>;
}

interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

export function LoadingSpinner({ size = 'sm', className = '' }: LoadingSpinnerProps) {
  const sizeClasses = {
    sm: 'w-3 h-3 border',
    md: 'w-4 h-4 border-2',
    lg: 'w-6 h-6 border-2'
  };
  
  return (
    <div 
      className={`${sizeClasses[size]} border-white/20 border-t-accent-cyan rounded-full animate-spin ${className}`}
    />
  );
}

