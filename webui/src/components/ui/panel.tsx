import * as React from 'react'

import { cn } from '@/lib/utils'

export function Panel({ className, ...props }: React.HTMLAttributes<HTMLElement>) {
  return (
    <section
      className={cn('rounded-md border border-[#d9d8d0] bg-[#fffffb] p-4 shadow-sm', className)}
      {...props}
    />
  )
}
