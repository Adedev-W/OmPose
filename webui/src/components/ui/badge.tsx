import * as React from 'react'
import { cva, type VariantProps } from 'class-variance-authority'

import { cn } from '@/lib/utils'

const badgeVariants = cva(
  'inline-flex h-7 items-center rounded-md px-2.5 text-xs font-medium',
  {
    variants: {
      variant: {
        default: 'bg-[#1f3d35] text-white',
        secondary: 'bg-[#e5eee8] text-[#1f3d35]',
        outline: 'border border-[#d9d8d0] bg-white text-[#514f48]',
      },
    },
    defaultVariants: {
      variant: 'default',
    },
  },
)

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

export function Badge({ className, variant, ...props }: BadgeProps) {
  return <div className={cn(badgeVariants({ variant, className }))} {...props} />
}
