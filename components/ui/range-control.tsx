type RangeControlProps = {
  id: string;
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  format?: (value: number) => string;
  onChange: (value: number) => void;
  help?: string;
};

export function RangeControl({
  id,
  label,
  value,
  min,
  max,
  step = 1,
  format = String,
  onChange,
  help,
}: RangeControlProps) {
  return (
    <div className="range-control">
      <div className="range-control__top">
        <label htmlFor={id}>{label}</label>
        <output htmlFor={id}>{format(value)}</output>
      </div>
      <input
        id={id}
        aria-describedby={help ? `${id}-help` : undefined}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
      {help ? (
        <small id={`${id}-help`} style={{ color: "var(--muted)" }}>
          {help}
        </small>
      ) : null}
    </div>
  );
}
